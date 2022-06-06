# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""
import copy
import torch
import torch.nn.functional as F

from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import (
    NATransformerModel, NATransformerDecoder, NATransformerEncoder, 
    FairseqNATEncoder, FairseqNATDecoder)
from fairseq.utils import new_arange, softmax, log_softmax
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.iterative_refinement_generator import DecoderOut

def _skeptical_unmasking(output_scores, output_masks, p=None, boundary_len=None):
    sorted_index = output_scores.sort(-1)[1]
    if boundary_len is None:
        boundary_len = (
            (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
        ).long()
    skeptical_mask = new_arange(output_masks).type_as(boundary_len) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


@register_model("cmlm_transformer")
class CMLMNATransformerModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.no_predict_length = getattr(args, "src_upsample", 1) > 1
        self.src_upsample = getattr(args, "src_upsample", 1)
        self.axe_eps_idx = encoder.dictionary.axe_eps_idx
        self.latent_dim = getattr(args, "latent_dim", 0)
        self.max_steps = getattr(args, "max_updates", 300000)
        self.ctc_bs_alpha = getattr(args, "ctc_bs_alpha", 0.)
        self.ctc_bs_beta = getattr(args, "ctc_bs_beta", 0.)
        self.use_ctc_beamsearch = getattr(args, "use_ctc_beamsearch", False)
        self.ctc_bs_beam = getattr(args, "ctc_bs_beam", 20)
        self.ctc_bs_lm_path = getattr(args, "ctc_bs_lm_path")

        if self.latent_dim > 0:
            args_copy = copy.deepcopy(args)
            args_copy.encoder_layers = getattr(args, "posterior_layers", 3)
            args_copy.decoder_layers = getattr(args, "posterior_layers", 3)
            args_copy.src_upsample = 1
            args_copy.length_loss_factor_tgt_len = False
            args_copy.use_first_token_pred_len = False
            args_copy.add_first_token_encoder = False
            args_copy.src_embedding_copy = None
            self.q_encoder_y = NATransformerEncoder(args_copy, self.decoder.dictionary, self.decoder.embed_tokens)
            self.q_encoder_xy = NATransformerDecoder(args_copy, self.encoder.dictionary, self.encoder.embed_tokens)
            self.q_encoder_xy.embed_length = None
            self.prob_esitmator = torch.nn.Linear(args.encoder_embed_dim, self.latent_dim * 2)
            self.latent_map = torch.nn.Linear(self.latent_dim, args.encoder_embed_dim)
            
            from fairseq.models.nat.levenshtein_utils import VAEBottleneck
            self.bottleneck = VAEBottleneck(
                args.encoder_embed_dim, 
                z_size=self.latent_dim, 
                freebits=getattr(args, "freebits", 1.0))

        if self.use_ctc_beamsearch:
            from ctcdecode import CTCBeamDecoder
            self.ctcdecoder = CTCBeamDecoder(
                self.tgt_dict.symbols,
                model_path=self.ctc_bs_lm_path,
                alpha=self.ctc_bs_alpha,
                beta=self.ctc_bs_beta,
                cutoff_top_n=40,
                cutoff_prob=1,
                beam_width=self.ctc_bs_beam,
                num_processes=20,
                blank_id=self.axe_eps_idx,
                log_probs_input=True
            )

    def set_num_updates(self, updates):
        self._updates = updates
        super().set_num_updates(updates)

    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
        parser.add_argument("--src-upsample", type=float,
                            help="if larger than 1, no length-prediction is used. upsample the source")
        # parser.add_argument("--src-upsample-bias", type=float, default=None)
        parser.add_argument("--dynamic-upsample", action='store_true')
        parser.add_argument("--replace-target-embed", action='store_true')
        parser.add_argument('--glat-sampling-ratio', type=float, default=0.0,
                            help='if larger than 0, use GLAT sampling.')
        parser.add_argument('--glat-min-ratio', type=float, default=None)
        parser.add_argument('--glat-use-valid', action='store_true')
        parser.add_argument('--poisson-mask', action='store_true')
        parser.add_argument("--ctc-bs-alpha", type=float, default=0.0)
        parser.add_argument("--ctc-bs-beam", type=int, default=20)
        parser.add_argument("--ctc-bs-beta", type=float, default=0.0)
        parser.add_argument('--use-ctc-beamsearch', action='store_true')
        parser.add_argument('--ctc-bs-lm-path', default=None)
        parser.add_argument('--glat-edit-distance', action='store_true')
        parser.add_argument('--glat-random-distance', action='store_true')
        parser.add_argument('--two-passes-distill', action='store_true')
        parser.add_argument('--two-passes-kl', action='store_true')

        parser.add_argument('--latent-dim', type=int, default=0)
        parser.add_argument('--posterior-layers', type=int, default=3)
        parser.add_argument('--simple-prior', action='store_true')
        parser.add_argument('--ar-prior', action='store_true')
        parser.add_argument('--freebits', type=float, default=1.0)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = CMLMNATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = CMLMNATransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder


    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        all_results = dict()

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        encoder_out = encoder_out._replace(
            encoder_embedding=encoder_out.encoder_out[1:].transpose(0, 1) \
                if (not self.no_predict_length) and getattr(self.args, "add_first_token_encoder", False) else \
                encoder_out.encoder_out.transpose(0, 1))

        # VAE part
        if self.latent_dim > 0:
            prior_prob = self.prob_esitmator(encoder_out.encoder_embedding)
                
            sampled_z, q_prob = self.bottleneck(
                self.q_encoder_xy.extract_features(src_tokens, 
                    encoder_out=self.q_encoder_y(tgt_tokens, src_lengths=tgt_tokens.ne(self.pad).sum(-1)))[0], 
                sampling=True)
            full_z = self.latent_map(sampled_z)
            encoder_out = encoder_out._replace(encoder_embedding=
                encoder_out.encoder_embedding + full_z)   # For simple, add Z on encoder out..
            kl, budget = self.bottleneck.compute_final_loss(q_prob, prior_prob, sampled_z)
            kl_loss = kl[src_tokens.ne(self.pad)].mean()
            all_results["kl_loss"] = {
                "loss": kl_loss, 'factor': 1.0,
            }
            all_results["add_logs"] = {'kl_budget': budget}
            
        # GLAT stuff
        if self.training and (getattr(self.args, "glat_sampling_ratio", 0) > 0):

            # go through pass-1 NAT
            if not getattr(self.args, "glat_use_valid", False):
                with torch.no_grad():
                    nat_word_ins_out, extra = self.decoder(
                        normalize=False,
                        prev_output_tokens=prev_output_tokens if not self.no_predict_length else None,
                        encoder_out=encoder_out)
            else:
                self.eval()  # disable all dropout
                with torch.no_grad():
                    glat_encoder_out = self.forward_encoder([src_tokens, src_lengths])
                    prev_decoder_out = self.initialize_output_tokens(glat_encoder_out, src_tokens)
                    if prev_decoder_out.attn is not None:
                        full_z = prev_decoder_out.attn
                        glat_encoder_out = glat_encoder_out._replace(encoder_embedding=
                            glat_encoder_out.encoder_embedding + full_z)   # For simple, add Z on encoder out..
                    nat_word_ins_out, extra = self.decoder(
                        normalize=False,
                        prev_output_tokens=prev_output_tokens if not self.no_predict_length else None,
                        encoder_out=glat_encoder_out,
                    )
                self.train()  # back to normal

            # apply GLAT?
            if (getattr(self.args, "glat_sampling_ratio", 0) > 0):
                # compute hamming distance
                f_ratio = self.args.glat_sampling_ratio
                if getattr(self.args, "glat_min_ratio", None) is not None:
                    f_min_ratio = self.args.glat_min_ratio
                    f_ratio = f_ratio - (f_ratio - f_min_ratio) * (self._updates / float(self.max_steps))
                    
                output_scores, output_tokens = nat_word_ins_out.max(-1)

                if self.no_predict_length:
                    from seqdist import ctc
                    alignment = ctc.viterbi_alignments(
                        log_softmax(nat_word_ins_out, dim=-1).transpose(0, 1), 
                        tgt_tokens, 
                        (~extra['padding_mask']).sum(-1), 
                        tgt_tokens.ne(self.pad).sum(-1))
                    
                    inter_tgt_tokens = ctc.interleave_blanks(tgt_tokens, self.axe_eps_idx)
                    aligned_tokens = torch.einsum('lbd,bd->bl', alignment, inter_tgt_tokens.float()).long()
                else:
                    inter_tgt_tokens = None
                    aligned_tokens = tgt_tokens.clone()

                # edit distance
                if getattr(self.args, "glat_edit_distance", False):
                    assert not getattr(self.args, "glat_random_distance", False)
                    from fairseq.models.nat.levenshtein_utils import _collapse, _get_edit_distance
                    out_tokens, _ = _collapse(output_tokens, output_scores, self.pad, self.axe_eps_idx)
                    edit_dis = _get_edit_distance(out_tokens, tgt_tokens, self.pad)
                    wer = edit_dis.type_as(output_scores) / tgt_tokens.ne(self.pad).sum(-1).type_as(output_scores)
                    mask_lens = f_ratio * wer * aligned_tokens.ne(self.pad).sum(-1).type_as(output_scores)
                # random
                elif getattr(self.args, "glat_random_distance", False):
                    assert not getattr(self.args, "glat_edit_distance", False)
                    bsz = aligned_tokens.size(0)
                    random_score = aligned_tokens.new_zeros((bsz,)).float().uniform_()
                    mask_lens = random_score * aligned_tokens.ne(self.pad).sum(1)
                # hamming
                else:
                    hamming_dis = output_tokens.ne(aligned_tokens).sum(-1)
                    mask_lens = (f_ratio * hamming_dis.type_as(output_scores))
                    

                decoder_scores = output_scores.uniform_().masked_fill_(
                        aligned_tokens.eq(self.pad) | aligned_tokens.eq(self.bos) | aligned_tokens.eq(self.eos), 2.0)
                if getattr(self.args, "poisson_mask", False):
                    mask_lens = torch.poisson(mask_lens)
                else:
                    mask_lens = mask_lens.long() 
                mask_lens = (aligned_tokens.ne(self.pad).sum(1) - mask_lens).clamp(min=1)
                glat_mask_ratio = mask_lens.sum() / float(aligned_tokens.ne(self.pad).sum())
                glat_mask = _skeptical_unmasking(decoder_scores, aligned_tokens.ne(self.pad), boundary_len=mask_lens[:, None])
                prev_output_tokens = aligned_tokens.masked_fill(glat_mask, self.unk)

                if "add_logs" not in all_results:
                    all_results["add_logs"] = {'glat_mask': glat_mask_ratio, 'f_ratio': f_ratio}
                else:
                    all_results["add_logs"]['glat_mask'] = glat_mask_ratio
                    all_results["add_logs"]['f_ratio'] = f_ratio

        # length prediction
        if not self.no_predict_length:
            length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
            length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)
            all_results["length"] = {
                "out": length_out, "tgt": length_tgt,
                "factor": self.decoder.get_length_loss_factor(length_out, tgt_tokens.ne(self.pad))
            }
            prev_output_tokens = (prev_output_tokens, )
        
        # decoding
        word_ins_out, extra = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out)

        factor = 1.

        if not self.no_predict_length:   # use the ground-truth length
            word_ins_mask = prev_output_tokens[0].eq(self.unk)
            word_ins_out = word_ins_out[:, :word_ins_mask.size(1)]
            out_mask = prev_output_tokens[0].ne(self.pad)
        else:
            word_ins_mask = None
            out_mask = ~extra['padding_mask']

        all_results["word_ins"] = {
            "out": word_ins_out, "tgt": tgt_tokens,
            "mask": word_ins_mask, "ls": self.args.label_smoothing,
            "nll_loss": True, "out_mask": out_mask, 'factor': factor,
        }
        if self.decoder.custom_loss is not None:
            all_results['decoder'] = self.decoder.custom_loss
        return all_results

    def initialize_output_tokens(self, encoder_out, src_tokens, tgt_tokens=None, length_tgt=None):
        # no length prediction. output dummy initialization.
        if self.latent_dim > 0:
            prior_prob = self.prob_esitmator(encoder_out.encoder_embedding)
            mean_vector = prior_prob[:, :, :self.latent_dim]
            full_z = self.latent_map(mean_vector)
        else:
            full_z = None

        if not self.no_predict_length:
            decoder_out = super().initialize_output_tokens(encoder_out, src_tokens, tgt_tokens, length_tgt)
            if full_z is None:
                return decoder_out
            return decoder_out._replace(attn=full_z)
        return DecoderOut(output_tokens=src_tokens, output_scores=None,
            attn=full_z, step=0, max_step=0, history=None)   # HACK: use attn for now..

    def regenerate_length_beam(self, decoder_out, beam_size, encoder_out):
        if not self.no_predict_length:
            return super().regenerate_length_beam(decoder_out, beam_size, encoder_out)

        b, l, d = decoder_out.attn.size()
        if self.latent_dim > 0:
            prior_prob = self.prob_esitmator(encoder_out.encoder_embedding)
            mean_vector = prior_prob[:, :, :self.latent_dim]
            var = 0.1 * F.softplus(prior_prob[:, :, self.latent_dim:])
            full_z = self.latent_map(mean_vector + var * torch.zeros_like(mean_vector).normal_())
        else:
            full_z = None
            
        return decoder_out._replace(
            output_tokens=decoder_out.output_tokens.unsqueeze(1).expand(b, beam_size, l).reshape(b * beam_size, l),
            attn=full_z,
        )
        # from fairseq import pdb; pdb.set_trace()

    def forward_encoder(self, encoder_inputs):
        encoder_out = self.encoder(*encoder_inputs)
        encoder_out = encoder_out._replace(
            encoder_embedding=encoder_out.encoder_out[1:].transpose(0, 1) \
                if (not self.no_predict_length) and getattr(self.args, "add_first_token_encoder", False) else \
                encoder_out.encoder_out.transpose(0, 1))
        return encoder_out

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):

        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history
        if decoder_out.attn is not None:
            full_z = decoder_out.attn
            encoder_out = encoder_out._replace(encoder_embedding=
                encoder_out.encoder_embedding + full_z)   # For simple, add Z on encoder out..

        # execute the decoder
        output_masks = output_tokens.eq(self.unk)
        decoder_output, extra = self.decoder(
            normalize=True,
            prev_output_tokens=(output_tokens,),
            encoder_out=encoder_out,
        )
        if self.use_ctc_beamsearch:
            topk = self.ctc_bs_beam  # * 2
            decoder_topk_scores, decoder_topk_index = decoder_output.topk(k=topk, dim=-1)

            # HACK: CTC beam-search requires the probability of blank, we put it in the end
            decoder_topk_scores = torch.cat([decoder_topk_scores, decoder_output[..., self.axe_eps_idx:self.axe_eps_idx+1]], -1)
            decoder_topk_index = torch.cat([decoder_topk_index, decoder_topk_index.new_ones(*decoder_topk_index.size()[:-1], 1) * self.axe_eps_idx], -1)
            if decoder_topk_index.size(0) > 1:
                decoder_topk_scores[..., 0].masked_fill_(extra["padding_mask"], 0.)
                decoder_topk_scores[..., -1].masked_fill_(extra["padding_mask"], 0.)
                decoder_topk_scores[..., 1:-1].masked_fill_(extra["padding_mask"].unsqueeze(-1), float("-Inf"))
                decoder_topk_index[...,0].masked_fill_(extra["padding_mask"], self.axe_eps_idx)

            beam_results, beam_scores, timesteps, out_lens = self.ctcdecoder.decode(decoder_topk_scores, decoder_topk_index)
            # from fairseq import pdb; pdb.set_trace()
            _scores, _tokens = beam_scores[:,0].to(decoder_output.device), beam_results[:, 0].to(decoder_output.device).long()
            out_lens = out_lens.to(decoder_output.device).type_as(_tokens)
            _scores = _scores[:, None].expand_as(_tokens)
            extra["padding_mask"] = new_arange(_tokens, *_tokens.size()) >= out_lens[:, :1]
        else:
            _scores, _tokens = decoder_output.max(-1)
        if not self.no_predict_length:
            output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
            output_scores.masked_scatter_(output_masks, _scores[output_masks])
        else:
            output_tokens = _tokens.masked_fill(extra["padding_mask"], self.pad)
            output_scores = _scores.masked_fill(extra["padding_mask"], 0.)

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            assert not self.no_predict_length, "mask-predict only supports length prediction."
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            )

            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history
        )


class CMLMNATransformerDecoder(NATransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.src_upsample = getattr(args, "src_upsample", 1)
        # self.upsample_bias = getattr(args, "src_upsample_bias", None)
        self.dynamic = getattr(args, "dynamic_upsample", False)
        self.replace_target_embed = getattr(args, "replace_target_embed", False)
        if self.src_upsample > 1:
            if not self.dynamic:
                # assert self.upsample_bias is None, "only support no bias"
                self.upsampler = torch.nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim * self.src_upsample)
            self.embed_length = None  # do not use length prediction
        self.custom_loss = None

    def dynamic_upsample(self, x, mask):
        l = x.new_ones(x.size(1), x.size(0)) * self.src_upsample
        l = l.masked_fill(mask, 0)
        e = torch.cumsum(l, 1)
        c = e - l / 2
        t = e[:, -1].ceil().long()
        t = new_arange(t, t.max())[None, :].expand(l.size(0), -1)  # B x L2
        t_mask = t >= e[:, -1:]   # target padding mask
        w = -(t[:, None, :] - c[:, :, None]) ** 2 / 0.3
        w = w.float()
        w = w.masked_fill(mask.unsqueeze(-1), -10000.0)
        t_w = F.softmax(w, dim=1)   # B x L x L2
        t_x = torch.einsum('bst,sbd->btd', t_w.type_as(x), x)
        return t_x, t_mask, w

    def forward_embedding(self, prev_output_tokens, embedding_copy=None, encoder_out=None):
        if self.src_upsample > 1:  # use encoder upsample
            B, L, D = encoder_out.encoder_embedding.size()
            x = encoder_out.encoder_embedding.transpose(0, 1)  # use embedding here
            mask = encoder_out.encoder_padding_mask[..., :L]
            if self.dynamic:
                x, mask, _ = self.dynamic_upsample(x, mask)
            else:
                x = self.upsampler(x.transpose(0, 1)).reshape(B, -1, D)
                mask = mask.unsqueeze(-1).expand(B, L, self.src_upsample).reshape(B, -1)
            
            if self.args.decoder_learned_pos:
                x = x + self.embed_positions(mask.long())
            
            if self.replace_target_embed and self.training and prev_output_tokens is not None:
                assert prev_output_tokens.size(1) == x.size(1), "length must match"
                tgt_embed, _ = super().forward_embedding(prev_output_tokens)
                tgt_mask = prev_output_tokens.ne(self.unk).unsqueeze(-1).expand_as(x)
                x = x.masked_scatter(tgt_mask, tgt_embed[tgt_mask])        
            return x, mask

        if len(prev_output_tokens) > 1:
            # concat all inputs with different positions
            assert embedding_copy is None, "does not support copy embedding for two inputs."
            positions = torch.cat([self.embed_positions(torch.zeros_like(p)) for p in prev_output_tokens], 1)
            prev_output_tokens = torch.cat([p for p in prev_output_tokens], 1)
        else:
            positions = None
            prev_output_tokens = prev_output_tokens[0]
        return super().forward_embedding(prev_output_tokens, embedding_copy, encoder_out, positions=positions)


class CMLMNATransformerEncoder(NATransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
    
    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        if not isinstance(src_tokens, tuple):
            positions = None
            src_tokens = src_tokens
        elif len(src_tokens) > 1:
            positions = torch.cat([self.embed_positions(torch.zeros_like(s)) for s in src_tokens], 1)
            src_tokens = torch.cat([s for s in src_tokens], 1)
        else:
            positions = None
            src_tokens = src_tokens[0]
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if positions is None:
            if self.embed_positions is not None:
                x = embed + self.embed_positions(src_tokens)
        else:
            x = embed + positions
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        if self.add_first_token:
            x = torch.cat([self.add_embed[None, None, :].expand(x.size(0), 1, x.size(2)), x], 1)
            encoder_padding_mask = torch.cat([encoder_padding_mask.new_zeros(x.size(0), 1), encoder_padding_mask], 1)
        return x, embed, encoder_padding_mask


@register_model_architecture("cmlm_transformer", "cmlm_transformer")
def cmlm_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.ngram_predictor = getattr(args, "ngram_predictor", 1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture("cmlm_transformer", "cmlm_transformer_wmt_en_de")
def cmlm_wmt_en_de(args):
    cmlm_base_architecture(args)

@register_model_architecture("cmlm_transformer", "cmlm_transformer_ctc")
def cmlm_wmt_axe(args):
    args.src_upsample = getattr(args, "src_upsample", 3)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    cmlm_base_architecture(args)