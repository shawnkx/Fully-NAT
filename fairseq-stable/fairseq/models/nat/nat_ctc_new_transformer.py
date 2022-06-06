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

def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t


@register_model("nat_ctc_new_transformer")
class NATransformerModel(FairseqNATModel):

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        
        parser.add_argument("--src-embedding-copy", type=str, default=None,
                            choices=['uniform', 'pos-attn', 'len-attn'],
                            help="copy encoder word embeddings as the initial input of the decoder")
        parser.add_argument("--copy-encoder-out", action='store_true',
                            help="only works when src-embedding-copy is not None.")
        parser.add_argument("--upsample-ratio", type=int, default=2,
                            help="upsample ratio of input")

        

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        assert args.no_cross_attention is True
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

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out)

        return {
            "word_ins": {
                "out": word_ins_out, "tgt": tgt_tokens,
                "mask": None,
                "ls": self.args.label_smoothing,
                "nll_loss": True, 
                "out_mask": ~encoder_out.encoder_padding_mask
            }
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        # step = decoder_out.step
        # output_tokens = decoder_out.output_tokens
        # output_scores = decoder_out.output_scores
        # history = decoder_out.history
        history = None

        # execute the decoder
        output_masks = encoder_out.encoder_padding_mask
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=None,
            encoder_out=encoder_out,
            step=0,
        ).max(-1)
        _tokens[output_masks] = self.pad
        _scores[output_masks] = 0.
        # output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        # output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=_tokens,
            output_scores=_scores,
            attn=None,
            history=history
        )

    def initialize_output_tokens(self, encoder_out, src_tokens, tgt_tokens=None, length_tgt=None):
        return DecoderOut(
            output_tokens=src_tokens,
            output_scores=None,
            attn=None,
            step=0,
            max_step=0,
            history=None
        )
        

    def regenerate_length_beam(self, decoder_out, beam_size):
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
        self.upsample_ratio = args.upsample_ratio
        self.upsampler = torch.nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim * args.upsample_ratio)

    def forward_embedding(self, src_tokens):
        x, embed, encoder_padding_mask = super().forward_embedding(src_tokens)
        x = self.upsampler(x).reshape(x.size(0), -1, x.size(2))
        encoder_padding_mask = encoder_padding_mask.unsqueeze(-1).expand(-1, -1, self.upsample_ratio)
        encoder_padding_mask = encoder_padding_mask.reshape(encoder_padding_mask.size(0), -1)
        return x, embed, encoder_padding_mask


class NATransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=True):
        assert no_encoder_attn is True
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.embed_positions = None
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.encoder_embed_dim = args.encoder_embed_dim
        self.src_embedding_copy = getattr(args, "src_embedding_copy", None)
        self.copy_encoder_out = getattr(args, "copy_encoder_out", False)
        if self.src_embedding_copy == 'pos-attn':
            self.pos_attn = MultiheadAttention(
                self.encoder_embed_dim, 1, dropout=0.0, self_attention=False)

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, **unused):
        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=self.src_embedding_copy,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out

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
        # x, decoder_padding_mask = self.forward_embedding(
        #     prev_output_tokens, 
        #     encoder_out=encoder_out,
        #     embedding_copy=embedding_copy)
        x, decoder_padding_mask = encoder_out.encoder_out, encoder_out.encoder_padding_mask
        # B x T x C -> T x B x C
        # x = x.transpose(0, 1)
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

        return x, {"attn": attn, "inner_states": inner_states}

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
        x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask

    def forward_copying_source(self, src_embeds, src_masks, tgt_masks, embedding_copy):
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        
        if embedding_copy == 'uniform':
            mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
                ~tgt_masks, 0
            )
            copied_embedding = torch.gather(
                src_embeds,
                1,
                mapped_inputs.unsqueeze(-1).expand(
                    *mapped_inputs.size(), src_embeds.size(-1)
                ),
            )
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


@register_model_architecture(
    "nat_ctc_new_transformer", "nat_ctc_new_transformer"
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
    args.no_cross_attention = True
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