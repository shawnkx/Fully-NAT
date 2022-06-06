# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import Embedding
from fairseq.modules import MultiheadAttention
from fairseq.models.nat import (
    FairseqNATModel,
    FairseqNATDecoder,
    FairseqNATEncoder,
    ensemble_decoder
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params


def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


def _argmax(x, dim):
    return (x == x.max(dim, keepdim=True)[0]).type_as(x)


def _uniform_assignment(src_masks, tgt_masks):
    tgt_lens = torch.sum(~tgt_masks, dim=1)
    src_lens = torch.sum(~src_masks, dim=1)
    max_trg_len = tgt_masks.size(1)
    # steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    
    # max_trg_len
    index_t = torch.cumsum(~tgt_masks, dim=1).float() / tgt_lens.unsqueeze(1) * (src_lens - 1).unsqueeze(1)
    # index_t = utils.new_arange(trg_lens, max_trg_len).float()
    # index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len

    index_t = torch.round(index_t).long()
    # print(index_t.shape, steps.shape,index_t, src_masks.size(1), tgt_masks.size(1))
    return index_t


@register_model("nonautoregressive_transformer")
class NATransformerModel(FairseqNATModel):

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        parser.add_argument("--src-embedding-copy", type=str, default=None,
                            choices=['uniform', 'soft-copy', 'pos-attn', 'len-attn'],
                            help="copy encoder word embeddings as the initial input of the decoder")
        parser.add_argument("--copy-encoder-out", action='store_true',
                            help="only works when src-embedding-copy is not None.")
        parser.add_argument("--pred-length-offset", action="store_true",
                            help="predicting the length difference between the target and source sentences")
        parser.add_argument("--sg-length-pred", action="store_true",
                            help="stop the gradients back-propagated from the length predictor")
        
        parser.add_argument("--length-loss-factor", type=float,
                            help="weights on the length prediction loss.")
        parser.add_argument("--length-loss-factor-tgt-len", action='store_true',
                            help="weights on the length prediction loss, divided by the target length")
        
        parser.add_argument("--add-first-token-encoder", action="store_true",
                            help="add additional first token to the encoder regardless left-pad or not.")
        parser.add_argument("--use-first-token-pred-len", action="store_true",
                            help="use first token to predict target sequence length")

        parser.add_argument("--pertub-gloden-length-delta", type=int,
                            help="if larger than 0, we will pertub the golden length")
        parser.add_argument("--use-predicted-length", action="store_true",
                            help="use predicted length instead of using gloden length")

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = NATransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        if getattr(self.args, "pertub_gloden_length_delta", 0) > 0:
            assert not self.args.pred_length_offset, "does not support offset"
            len_delta = self.args.pertub_gloden_length_delta
            len_delta = torch.randint(low=-len_delta, high=len_delta, 
                size=(length_out.size(0),), device=length_out.device)
            length_pred = (length_tgt + len_delta).clamp(min=3 if self.args.prepend_bos else 2)
            prev_output_tokens = self.initialize_output_tokens(
                encoder_out, src_tokens, length_tgt=length_pred).output_tokens
    
        if getattr(self.args, "use_predicted_length", False):
            assert self.args.pred_length_offset, "only support offset"
            length_pred = self.decoder.forward_length_prediction(length_out, encoder_out)
            prev_output_tokens = self.initialize_output_tokens(
                encoder_out, src_tokens, length_tgt=length_pred).output_tokens

        # decoding
        word_ins_out, _ = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out)

        return {
            "word_ins": {
                "out": word_ins_out, "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad), "ls": self.args.label_smoothing,
                "nll_loss": True, "out_mask": prev_output_tokens.ne(self.pad)
            },
            "length": {
                "out": length_out, "tgt": length_tgt,
                "factor": self.decoder.get_length_loss_factor(length_out, tgt_tokens.ne(self.pad))
            }
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        )[0].max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history
        )

    def initialize_output_tokens(self, encoder_out, src_tokens, tgt_tokens=None, length_tgt=None):
        if length_tgt is None:
            # length prediction
            length_tgt = self.decoder.forward_length_prediction(
                self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
                encoder_out=encoder_out,
                tgt_tokens=tgt_tokens
            )

        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        if self.args.prepend_bos:
            initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out.encoder_out)

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None
        )

    def regenerate_length_beam(self, decoder_out, beam_size, *args, **kwargs):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = length_tgt[:, None] + utils.new_arange(length_tgt, 1, beam_size) - beam_size // 2
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores
        )


class NATransformerEncoder(FairseqNATEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.add_first_token = getattr(args, "add_first_token_encoder", False)
        if self.add_first_token:
            self.add_embed = torch.nn.Parameter(
                torch.zeros(args.encoder_embed_dim).normal_(mean=0.0, std=0.02),
                requires_grad=True)

    def forward_embedding(self, src_tokens):
        x, embed, encoder_padding_mask = super().forward_embedding(src_tokens)
        if self.add_first_token:
            x = torch.cat([self.add_embed[None, None, :].expand(x.size(0), 1, x.size(2)), x], 1)
            encoder_padding_mask = torch.cat([encoder_padding_mask.new_zeros(x.size(0), 1), encoder_padding_mask], 1)
        return x, embed, encoder_padding_mask


class NATransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.encoder_embed_dim = args.encoder_embed_dim
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.use_len_token = getattr(args, "use_first_token_pred_len", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.src_embedding_copy = getattr(args, "src_embedding_copy", None)
        self.copy_encoder_out = getattr(args, "copy_encoder_out", False)

        if self.src_embedding_copy == 'pos-attn':
            self.pos_attn = MultiheadAttention(
                self.encoder_embed_dim, 1, dropout=0.0, self_attention=False)
        elif self.src_embedding_copy == 'soft-copy':
            self.tau = torch.nn.Parameter(torch.scalar_tensor(0.5), requires_grad=True)

        self.embed_length = Embedding(1024, self.encoder_embed_dim, None)
        if self.pred_length_offset:
            self.bias_length = torch.zeros(1024)
            self.bias_length[512] = 3.  # add bias to predict source length
            self.bias_length = torch.nn.Parameter(self.bias_length, requires_grad=True)

        self._length_loss_factor = getattr(args, "length_loss_factor", 0.1)
        self._length_factor_tgtl = getattr(args, "length_loss_factor_tgt_len", False)

        if getattr(args, "use_first_token_pred_len", False) and \
            (not getattr(args, "add_first_token_encoder", False)):
            
            assert (not args.left_pad_source) and args.prepend_bos, (
                "arguments left-source-pad must be False and prepend-bos must be True"
                "for using the first token to predict the target sequence length"
                )

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=self.src_embedding_copy,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out, extra

    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        if self.use_len_token:
            enc_feats = encoder_out.encoder_out[0] # B X C
        else:
            enc_feats = encoder_out.encoder_out  # T x B x C
            src_masks = encoder_out.encoder_padding_mask  # B x T or None
            enc_feats = _mean_pooling(enc_feats, src_masks)
        
        if self.sg_length_pred:
            enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        
        if self.pred_length_offset:
            length_out = length_out + self.bias_length[None, :]
        
        return F.log_softmax(length_out, -1) if normalize else length_out

    def get_length_loss_factor(self, length_out=None, masks=None):
        if not self._length_factor_tgtl:
            return self._length_loss_factor
        return length_out.size(0) / masks.sum().type_as(length_out)

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=None,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        x, decoder_padding_mask = self.forward_embedding(
            prev_output_tokens, 
            encoder_out=encoder_out,
            embedding_copy=embedding_copy)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states, "padding_mask": decoder_padding_mask}

    def forward_embedding(self, prev_output_tokens, embedding_copy=None, encoder_out=None, positions=None, **unused):
        if embedding_copy:
            assert encoder_out.encoder_embedding is not None
            if self.copy_encoder_out:
                src_lens = encoder_out.encoder_embedding.size(1)
                src_embd = encoder_out.encoder_out.transpose(0, 1)
                src_embd = src_embd[:, :src_lens]
                src_mask = encoder_out.encoder_padding_mask[:, :src_lens]
            else:
                src_embd = encoder_out.encoder_embedding
                src_mask = encoder_out.encoder_padding_mask
            
            if self.use_len_token:
                src_mask = src_mask[:, 1:]
                if src_mask.size(1) < src_embd.size(1):
                    src_embd = src_embd[:, 1:]

            states = self.forward_copying_source(
                    src_embd, src_mask, 
                    prev_output_tokens.eq(self.padding_idx),   # pad mask
                    embedding_copy
                )
        else:
            states = None
            
        # embed positions
        if positions is None:
            positions = (
                self.embed_positions(prev_output_tokens)
                if self.embed_positions is not None
                else None
            )

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        if states is not None:
            unk_mask = prev_output_tokens.eq(self.unk).type_as(x).unsqueeze(-1)
            x = states * unk_mask + x * (1 - unk_mask)
        
        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask

    def forward_copying_source(self, src_embeds, src_masks, tgt_masks, embedding_copy):
        length_sources = (~src_masks).sum(1)
        length_targets = (~tgt_masks).sum(1)
        
        if embedding_copy == 'uniform':
            mapped_inputs = _uniform_assignment(src_masks, tgt_masks).masked_fill(
                tgt_masks, 0
            )
            # from fairseq import pdb; pdb.set_trace()
            copied_embedding = torch.gather(
                src_embeds,
                1,
                mapped_inputs.unsqueeze(-1).expand(
                    *mapped_inputs.size(), src_embeds.size(-1)
                ),
            )
        elif embedding_copy == 'soft-copy':
            # compute the distance matrix
            src_pos = torch.cumsum(~src_masks, dim=1).unsqueeze(-1).expand(-1, -1, tgt_masks.size(1))
            tgt_pos = torch.cumsum(~tgt_masks, dim=1).unsqueeze(1).expand(-1, src_masks.size(1), -1)
            dist_ij = -torch.abs(src_pos - tgt_pos) / self.tau
            dist_ij = dist_ij.masked_fill(src_masks[:, :, None], float("-inf"))
            w_ij = F.softmax(dist_ij, dim=1)   # B x S x T
            copied_embedding = torch.einsum('bst,bsd->btd', w_ij, src_embeds)
 
        elif embedding_copy == 'pos-attn':
            pos_inputs = self.embed_positions(tgt_masks.long())
            attn_outputs, _ = self.pos_attn(
                query=pos_inputs.transpose(0, 1),
                key=src_embeds.transpose(0, 1),
                value=src_embeds.transpose(0, 1),
                key_padding_mask=src_masks
            )
            copied_embedding = attn_outputs.transpose(0, 1)
        else:
            raise NotImplementedError
        return copied_embedding

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None):
        enc_feats = encoder_out.encoder_out  # T x B x C
        src_masks = encoder_out.encoder_padding_mask  # B x T or None
        if self.pred_length_offset:
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(
                    enc_feats.size(0)
                )
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            if self.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + 512
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=1023)  # BUG

        else:
            # predict the length target (greedy for now)
            # TODO: implementing length-beam
            pred_lengs = length_out.max(-1)[1]
            if self.pred_length_offset:
                length_tgt = pred_lengs - 512 + src_lengs
            else:
                length_tgt = pred_lengs
        return length_tgt


@register_model_architecture(
    "nonautoregressive_transformer", "nonautoregressive_transformer"
)
def base_architecture(args):
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
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture(
    "nonautoregressive_transformer", "nonautoregressive_transformer_wmt_en_de"
)
def nonautoregressive_transformer_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture("nonautoregressive_transformer", "nat_axe")
def nat_wmt_axe(args):
    args.length_loss_factor_tgt_len = getattr(args, "length_loss_factor_tgt_len", True)
    args.use_first_token_pred_len = getattr(args, "use_first_token_pred_len", True)
    args.add_first_token_encoder = getattr(args, "add_first_token_encoder", True)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_architecture(args)