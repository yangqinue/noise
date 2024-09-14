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

import torch
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nonautoregressive_transformer import NATransformerModel


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = (
        torch.arange(output_masks.size(1), device=output_masks.device)[None, :]
        < boundary_len
    )
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


@register_model("cmlm_transformer")
class CMLMNATransformerModel(NATransformerModel):
    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        length_out, length_tgt = self.decoder.forward_length_prediction(
            encoder_out, tgt_tokens
        )

        word_ins_out, word_ins_tgt, _ = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, tgt_tokens=tgt_tokens
        )
        word_ins_mask = prev_output_tokens.eq(self.unk)
        return {
            "word_ins_out": word_ins_out,
            "word_ins_tgt": word_ins_tgt,
            "word_ins_mask": word_ins_mask,
            "length_out": length_out,
            "length_tgt": length_tgt,
            "length_w": self.decoder.length_loss_factor,
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):

        step = decoder_out["step"]
        max_step = decoder_out["max_step"]

        output_tokens = decoder_out["output_tokens"]
        output_scores = decoder_out["output_scores"]

        # execute the decoder
        output_masks = output_tokens.eq(self.unk)
        _scores, _tokens = self.decoder(
            output_tokens, encoder_out=encoder_out, decoding_format=decoding_format
        )
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            )

            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

        return {"output_tokens": output_tokens, "output_scores": output_scores}


@register_model_architecture("cmlm_transformer", "cmlm_transformer")
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
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
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
def iter_nat_wmt_en_de(args):
    base_architecture(args)
