from functools import reduce

import torch


def _hash(data_iter):
    MAGIC1 = 131313
    MAGIC2 = 10000003
    hash_score = 0
    for data in data_iter:
        if data is None:
            return None
        hash_score = ((MAGIC1 * hash_score) + data) % MAGIC2
    return hash_score


def _ngram(data, n):
    length = data.size(1)
    if n > length:
        return [None]

    for ni in range(n):
        yield data[:, ni:length - n + ni + 1]


def _matching(D1, D2, M1, M2):
    S = D1.unsqueeze(-1) == D2.unsqueeze(-2)
    M = M1.unsqueeze(-1) & M2.unsqueeze(-2)
    return (S & M).max(-1)[0].sum(-1)


def _count(D1, D2, M1, M2, n):
    if D1.size(-1) < n:
        return D1.new_zeros(D1.size(0)), D1.new_zeros(D1.size(0))
    return _matching(
        _hash(_ngram(D1, n)), _hash(_ngram(D2, n)), M1[:, n - 1:],
        M2[:, n - 1:]), M1[:, n - 1:].sum(-1)


def _div(m, c):
    return m / (c + 1e-9)


def brevity(reflen, predlen):
    return torch.exp(1 - reflen / (predlen + 1e-9)).clamp(max=1)


def batched_bleu_score(ref,
                       hyp,
                       ref_mask=None,
                       hyp_mask=None,
                       smooth=False,
                       corpus=False,
                       remove_lp=False,
                       dtype=torch.float32,
                       ignores=[0, 1, 2, 3]):
    if ref_mask is None:
        ref_mask = reduce(lambda x, y: x & y, [ref != k for k in ignores])
    if hyp_mask is None:
        hyp_mask = reduce(lambda x, y: x & y, [hyp != k for k in ignores])

    ref_len = ref_mask.sum(-1).to(dtype)
    hyp_len = hyp_mask.sum(-1).to(dtype)

    if corpus:
        ref_len = ref_len.sum(0)
        hyp_len = hyp_len.sum(0)

    scores = 0
    for n in range(1, 5):
        try:
            m, c = _count(hyp, ref, hyp_mask, ref_mask, n)
        except AttributeError:
            m, c = hyp.new_zeros(hyp.size(0)), hyp.new_zeros(hyp.size(0))

        if smooth and (n > 1):
            m += 1
            c += 1
        if corpus:
            m = m.sum(0)
            c = c.sum(0)
        scores += torch.log(_div(m.to(dtype), c.to(dtype)))

    if remove_lp:
        return torch.exp(scores / 4)
    return brevity(ref_len, hyp_len) * torch.exp(scores / 4)
