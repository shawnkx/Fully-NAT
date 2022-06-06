# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from fairseq.utils import new_arange

from numba import jit
from torch.autograd import Function
from numba import cuda

# -------------- Helper Functions --------------------------------------------------- #

def load_libnat():
    try:
        from fairseq import libnat_cuda
        return libnat_cuda, True

    except ImportError as e:
        print(str(e) + '... fall back to CPU version')

        try:
            from fairseq import libnat
            return libnat, False

        except ImportError as e:
            import sys
            sys.stderr.write("ERROR: missing libnat_cuda. run `python setup.py build_ext --inplace`\n")
            raise e

def _get_edit_distance(in_tokens, out_tokens, padding_idx):
    libnat, use_cuda = load_libnat()
    return libnat.levenshtein_distance3(
        in_tokens.int(), out_tokens.int(),
        in_tokens.ne(padding_idx).sum(-1).int(),
        out_tokens.ne(padding_idx).sum(-1).int())


def _get_ins_targets(in_tokens, out_tokens, padding_idx, unk_idx):
    libnat, use_cuda = load_libnat()

    def _get_ins_targets_cuda(in_tokens, out_tokens, padding_idx, unk_idx):
        in_masks = in_tokens.ne(padding_idx)
        out_masks = out_tokens.ne(padding_idx)
        mask_ins_targets, masked_tgt_masks = libnat.generate_insertion_labels(
            out_tokens.int(), libnat.levenshtein_distance(
                in_tokens.int(), out_tokens.int(),
                in_masks.sum(1).int(), out_masks.sum(1).int()
            )
        )
        masked_tgt_masks = masked_tgt_masks.bool() & out_masks
        mask_ins_targets = mask_ins_targets.type_as(
            in_tokens)[:, 1:in_masks.size(1)].masked_fill_(~in_masks[:, 1:], 0)
        masked_tgt_tokens = out_tokens.masked_fill(masked_tgt_masks, unk_idx)
        return masked_tgt_masks, masked_tgt_tokens, mask_ins_targets

    def _get_ins_targets_cpu(in_tokens, out_tokens, padding_idx, unk_idx):
        in_seq_len, out_seq_len = in_tokens.size(1), out_tokens.size(1)

        in_tokens_list = [
            [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
        ]
        out_tokens_list = [
            [t for t in s if t != padding_idx]
            for i, s in enumerate(out_tokens.tolist())
        ]

        full_labels = libnat.suggested_ed2_path(
            in_tokens_list, out_tokens_list, padding_idx
        )
        mask_inputs = [
            [len(c) if c[0] != padding_idx else 0 for c in a[:-1]] for a in full_labels
        ]

        # generate labels
        masked_tgt_masks = []
        for mask_input in mask_inputs:
            mask_label = []
            for beam_size in mask_input[1:-1]:  # HACK 1:-1
                mask_label += [0] + [1 for _ in range(beam_size)]
            masked_tgt_masks.append(
                mask_label + [0 for _ in range(out_seq_len - len(mask_label))]
            )
        mask_ins_targets = [
            mask_input[1:-1] + [0 for _ in range(in_seq_len - 1 - len(mask_input[1:-1]))]
            for mask_input in mask_inputs
        ]

        # transform to tensor
        masked_tgt_masks = torch.tensor(
            masked_tgt_masks, device=out_tokens.device
        ).bool()
        mask_ins_targets = torch.tensor(mask_ins_targets, device=in_tokens.device)
        masked_tgt_tokens = out_tokens.masked_fill(masked_tgt_masks, unk_idx)
        return masked_tgt_masks, masked_tgt_tokens, mask_ins_targets

    if use_cuda:
        return _get_ins_targets_cuda(in_tokens, out_tokens, padding_idx, unk_idx)
    return _get_ins_targets_cpu(in_tokens, out_tokens, padding_idx, unk_idx)


def _get_del_targets(in_tokens, out_tokens, padding_idx):
    libnat, use_cuda = load_libnat()

    def _get_del_targets_cuda(in_tokens, out_tokens, padding_idx):
        in_masks = in_tokens.ne(padding_idx)
        out_masks = out_tokens.ne(padding_idx)

        word_del_targets = libnat.generate_deletion_labels(
            in_tokens.int(),
            libnat.levenshtein_distance(
                in_tokens.int(), out_tokens.int(),
                in_masks.sum(1).int(), out_masks.sum(1).int()
            )
        )
        word_del_targets = word_del_targets.type_as(in_tokens).masked_fill_(~in_masks, 0)
        return word_del_targets

    def _get_del_targets_cpu(in_tokens, out_tokens, padding_idx):
        out_seq_len = out_tokens.size(1)
        with torch.cuda.device_of(in_tokens):
            in_tokens_list = [
                [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
            ]
            out_tokens_list = [
                [t for t in s if t != padding_idx]
                for i, s in enumerate(out_tokens.tolist())
            ]

        full_labels = libnat.suggested_ed2_path(
            in_tokens_list, out_tokens_list, padding_idx
        )
        word_del_targets = [b[-1] for b in full_labels]
        word_del_targets = [
            labels + [0 for _ in range(out_seq_len - len(labels))]
            for labels in word_del_targets
        ]

        # transform to tensor
        word_del_targets = torch.tensor(word_del_targets, device=out_tokens.device)
        return word_del_targets

    if use_cuda:
        return _get_del_targets_cuda(in_tokens, out_tokens, padding_idx)
    return _get_del_targets_cpu(in_tokens, out_tokens, padding_idx)


def _apply_ins_masks(
    in_tokens, in_scores, mask_ins_pred, padding_idx, unk_idx, eos_idx
):

    in_masks = in_tokens.ne(padding_idx)
    in_lengths = in_masks.sum(1)

    # HACK: hacky way to shift all the paddings to eos first.
    in_tokens.masked_fill_(~in_masks, eos_idx)
    mask_ins_pred.masked_fill_(~in_masks[:, 1:], 0)

    out_lengths = in_lengths + mask_ins_pred.sum(1)
    out_max_len = out_lengths.max()
    out_masks = (
        new_arange(out_lengths, out_max_len)[None, :]
        < out_lengths[:, None]
    )

    reordering = (mask_ins_pred + in_masks[:, 1:].long()).cumsum(1)
    out_tokens = (
        in_tokens.new_zeros(in_tokens.size(0), out_max_len)
        .fill_(padding_idx)
        .masked_fill_(out_masks, unk_idx)
    )
    out_tokens[:, 0] = in_tokens[:, 0]
    out_tokens.scatter_(1, reordering, in_tokens[:, 1:])

    out_scores = None
    if in_scores is not None:
        in_scores.masked_fill_(~in_masks, 0)
        out_scores = in_scores.new_zeros(*out_tokens.size())
        out_scores[:, 0] = in_scores[:, 0]
        out_scores.scatter_(1, reordering, in_scores[:, 1:])

    return out_tokens, out_scores


def _apply_ins_words(
    in_tokens, in_scores, word_ins_pred, word_ins_scores, unk_idx
):
    word_ins_masks = in_tokens.eq(unk_idx)
    out_tokens = in_tokens.masked_scatter(word_ins_masks, word_ins_pred[word_ins_masks])

    if in_scores is not None:
        out_scores = in_scores.masked_scatter(
            word_ins_masks, word_ins_scores[word_ins_masks]
        )
    else:
        out_scores = None

    return out_tokens, out_scores


def _apply_del_words(
    in_tokens, in_scores, in_attn, word_del_pred, padding_idx, bos_idx, eos_idx
):
    # apply deletion to a tensor
    in_masks = in_tokens.ne(padding_idx)
    bos_eos_masks = in_tokens.eq(bos_idx) | in_tokens.eq(eos_idx)

    max_len = in_tokens.size(1)
    word_del_pred.masked_fill_(~in_masks, 1)
    word_del_pred.masked_fill_(bos_eos_masks, 0)

    reordering = (
        new_arange(in_tokens)
        .masked_fill_(word_del_pred, max_len)
        .sort(1)[1]
    )

    out_tokens = in_tokens.masked_fill(word_del_pred, padding_idx).gather(1, reordering)

    out_scores = None
    if in_scores is not None:
        out_scores = in_scores.masked_fill(word_del_pred, 0).gather(1, reordering)

    out_attn = None
    if in_attn is not None:
        _mask = word_del_pred[:, :, None].expand_as(in_attn)
        _reordering = reordering[:, :, None].expand_as(in_attn)
        out_attn = in_attn.masked_fill(_mask, 0.).gather(1, _reordering)

    return out_tokens, out_scores, out_attn


def _skip(x, mask):
    """
    Getting sliced (dim=0) tensor by mask. Supporting tensor and list/dict of tensors.
    """
    if isinstance(x, int):
        return x

    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        if x.size(0) == mask.size(0):
            return x[mask]
        elif x.size(1) == mask.size(0):
            return x[:, mask]

    if isinstance(x, list):
        return [_skip(x_i, mask) for x_i in x]

    if isinstance(x, dict):
        return {k: _skip(v, mask) for k, v in x.items()}

    raise NotImplementedError


def _skip_encoder_out(encoder, encoder_out, mask):
    if not mask.any():
        return encoder_out
    else:
        return encoder.reorder_encoder_out(encoder_out, mask.nonzero().squeeze())


def _fill(x, mask, y, padding_idx):
    """
    Filling tensor x with y at masked positions (dim=0).
    """
    if x is None:
        return y
    assert x.dim() == y.dim() and mask.size(0) == x.size(0)
    assert x.dim() == 2 or (x.dim() == 3 and x.size(2) == y.size(2))
    n_selected = mask.sum()
    assert n_selected == y.size(0)

    if n_selected == x.size(0):
        return y

    if x.size(1) < y.size(1):
        dims = [x.size(0), y.size(1) - x.size(1)]
        if x.dim() == 3:
            dims.append(x.size(2))
        x = torch.cat([x, x.new_zeros(*dims).fill_(padding_idx)], 1)
        x[mask] = y
    elif x.size(1) > y.size(1):
        x[mask] = padding_idx
        if x.dim() == 2:
            x[mask, :y.size(1)] = y
        else:
            x[mask, :y.size(1), :] = y
    else:
        x[mask] = y
    return x


@cuda.jit
def collapse_repetition_cuda(tokens, scores, length, pad, remove_blank):
    b = cuda.blockIdx.x
    ct, cs, ci = -1, 0, -1
    s = 1.0
    for i in range(0, length):
        if tokens[b, i] == ct:
            scores[b, ci] += scores[b, i]
            s += 1.0
            continue
        ct = tokens[b, i]
        scores[b, ci] /= s
        s = 1.0
        if (remove_blank > -1) and (ct == remove_blank):
            continue
        ci += 1
        tokens[b, ci] = tokens[b, i]
        scores[b, ci] = scores[b, i]
        
    for i in range(ci + 1, length):
        tokens[b, i] = pad
        scores[b, i] = 0.0

@jit(nopython=True)
def collapse_repetition_cpu(tokens, scores, length, B, pad, remove_blank):
    for b in range(B):
        ct, cs, ci = -1, 0, -1
        for i in range(0, length):
            if tokens[b, i] == ct:
                continue
            ct = tokens[b, i]
            if (remove_blank > -1) and (ct == remove_blank):
                continue
            ci += 1
            tokens[b, ci] = tokens[b, i]
            scores[b, ci] = scores[b, i]
            
        for i in range(ci + 1, length):
            tokens[b, i] = pad
            scores[b, i] = 0.0

class _COLLAPSE(Function):
    @staticmethod
    def forward(ctx, tokens, scores, pad, remove_blank=-1):
        B, length = tokens.size(0), tokens.size(1)
        dtype = scores.dtype

        if scores.is_cuda:
            collapse_repetition_cuda[B, 1](
                cuda.as_cuda_array(tokens.detach()), 
                cuda.as_cuda_array(scores.float().detach()),
                length, pad, remove_blank)
        else:
            collapse_repetition_cpu(
                tokens.detach().numpy(), 
                scores.float().detach().numpy(),
                length, B, pad, remove_blank)
        scores = scores.type(dtype)
        return tokens, scores
    
_collapse = _COLLAPSE.apply

@cuda.jit
def ctc_beam_search(topk_tokens, topk_scores, hypo_tokens, hypo_lens, hypo_scores, lengths, pad, blank, beam_size):
    b = cuda.blockIdx.x    # batch id
    i = cuda.threadIdx.x   # hypo id
    
    def logsumexp(a, b):
        a_max = max(a, b)
        lsp = math.exp(a - a_max) + math.exp(b - a_max)
        lsp = math.log(lsp)
        return lsp + a_max
    
    # sorted(topk_scores[b, :, i])
    # topk_scores[b, :, i] = np.sort(topk_scores[b, :, i], axis=1)    
    # for t in range(lengths):
    #     if (i > 0) and (t == 0):
    #         pass # do nothing for other threads in the first step
    #     else:
    #         # for batch_id b, hypo i, timestep t,
    #         if (hypo_lens[b])

    #     hypo_tokens[b, t, i] = topk_tokens[b, t, 0]


class _CTCBEAMSEARCH(Function):
    @staticmethod
    def forward(ctx, log_probs, pad, blank, beam_size=10, cutoff=40):
        topk_scores, topk_tokens = log_probs.topk(k=cutoff, dim=-1)
        B, length = topk_tokens.size(0), topk_tokens.size(1)
        
        hypo_tokens = topk_tokens.new_zeros(B, length, beam_size)
        hypo_lens = topk_tokens.new_zeros(B, beam_size)
        hypo_scores = topk_scores.new_zeros(B, 2)
        hypo_scores[:, 1] = float("-inf")
        
        raise NotImplementedError

        ctc_beam_search[B, beam_size](
                cuda.as_cuda_array(topk_tokens.detach()), 
                cuda.as_cuda_array(topk_scores.float().detach()),
                cuda.as_cuda_array(hypo_tokens),
                cuda.as_cuda_array(hypo_lens),
                cuda.as_cuda_array(hypo_scores),        
                length, pad, blank, beam_size)
        
        from fairseq import pdb;pdb.set_trace()
_ctcdecode = _CTCBEAMSEARCH.apply


@torch.enable_grad()
def ctc_alignment_targets(logits, targets, input_lengths, target_lengths, blank):
    logits.requires_grad = True
    log_probs = log_softmax(logits, dim = -1)
    ctc_loss = F.ctc_loss(log_probs.transpose(0, 1), targets, input_lengths, target_lengths, blank = blank, reduction = 'sum')
    ctc_grad, = torch.autograd.grad(ctc_loss, (logits,), retain_graph = True)
    temporal_mask = (new_arange(input_lengths, logits.shape[1]).unsqueeze(0) < input_lengths.unsqueeze(1))[:, None, :]
    alignment_targets = (log_probs.exp() * temporal_mask.transpose(1, 2) - ctc_grad).detach()
    return alignment_targets


def gumbel_softmax_sample(logits, temperature, eps=1e-7):
    U = torch.rand(logits.size()).type_as(logits)
    sample = -torch.log(-torch.log(U + eps) + eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1,latent_dim*categorical_dim)


class VAEBottleneck(nn.Module):
    # copy from https://github.com/zomux/lanmt/blob/8099f03bb8df54cf22dabc811f3b3b5c63603101/lib_vae.py#L13
    def __init__(self, 
        hidden_size, z_size=None, max_steps=300000, freebits=1):
        super(VAEBottleneck, self).__init__()
        self.hidden_size = hidden_size

        if z_size is None:
            self.z_size = self.hidden_size
        else:
            self.z_size = z_size
        
        self.dense = nn.Linear(hidden_size, self.z_size * 2)

        self.eps = 1e-6
        self.KL_budget = freebits
        self.max_train_steps = max_steps # fixed for now
        self.budget_annealing = True

    def forward(self, x, sampling=True, residual_q=None):
        vec = self.dense(x)
        mu = vec[:, :, :self.z_size]
        if residual_q is not None:
            mu = 0.5 * (mu + residual_q[:, :, :self.z_size])
        if not sampling:
            return mu, vec
        else:
            var = F.softplus(vec[:, :, self.z_size:])
            if residual_q is not None:
                var = 0.5 * (var + F.softplus(residual_q[:, :, self.z_size:]))
            noise = mu.clone()
            noise = noise.normal_()
            z = mu + noise * var
            return z, vec
    def sample_any_dist(self, dist, deterministic=False, samples=1, noise_level=1.):
        mu = dist[:, :, :self.z_size]
        if deterministic:
            return mu
        else:
            var = F.softplus(dist[:, :, self.z_size:])
            noise = mu.clone()
            if samples > 1:
                if noise.shape[0] == 1:
                    noise = noise.expand(samples, -1, -1).clone()
                    mu = mu.expand(samples, -1, -1).clone()
                    var = var.expand(samples, -1, -1).clone()
                else:
                    noise = noise[:, None, :, :].expand(-1, samples, -1, -1).clone()
                    mu = mu[:, None, :, :].expand(-1, samples, -1, -1).clone()
                    var = var[:, None, :, :].expand(-1, samples, -1, -1).clone()

            noise = noise.normal_()
            z = mu + noise * var * noise_level
            return z
    
    def set_num_updates(self, updates):
        self._updates = updates

    def compute_vae_KL(self, prior_prob, q_prob, z):
        """Compute KL divergence given two Gaussians.
        """
        mu1 = q_prob[:, :, :self.z_size]
        var1 = F.softplus(q_prob[:, :, self.z_size:])  
        mu2 = prior_prob[:, :, :self.z_size]
        var2 = F.softplus(prior_prob[:, :, self.z_size:])
        kl = torch.log(var2 / (var1 + 1e-8) + 1e-8) + (
                    (torch.pow(var1, 2) + torch.pow(mu1 - mu2, 2)) / (2 * torch.pow(var2, 2))) - 0.5
        kl = kl.sum(-1)
        return kl

    def compute_final_loss(self, q_prob, prior_prob, z=None):
        """ Compute the report the loss.
        """
        kl = self.compute_vae_KL(prior_prob, q_prob, z)

        # Apply budgets for KL divergence: KL = max(KL, budget)
        budget_upperbound = self.KL_budget
        if self.budget_annealing:
            step = self._updates
            half_maxsteps = min(float(self.max_train_steps / 2), 18000) / 2
            if step > half_maxsteps:
                rate = (float(step) - half_maxsteps) / half_maxsteps
                min_budget = 0.
                budget = min_budget + (budget_upperbound - min_budget) * (1. - rate)
            else:
                budget = budget_upperbound
        else:
            budget = self.KL_budget

        # Compute KL divergence
        max_mask = ((kl - budget) > 0.).type_as(kl)
        kl = kl * max_mask + (1. - max_mask) * budget
        
        return kl, budget
