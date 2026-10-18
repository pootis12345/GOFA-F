"""Stage19 canonical whole-answer tokens and a single finite numeric alphabet."""
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F

from gofa_f.allnode_codec import AllNodeCodec, OFFICIAL_PROMPT


class ShortCodec(AllNodeCodec):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        vocab = tokenizer.get_vocab()
        self.finite_words = tuple(f"###{x:.4f}###" for x in np.unique(self.centers))
        self.finite_ids = tuple(vocab.get(w) for w in self.finite_words)
        if len(self.finite_ids) != 10000 or None in self.finite_ids or len(set(self.finite_ids)) != 10000:
            raise ValueError("Finite alphabet must have exactly 10000 unique numeric tokens")
        self.nan_id = vocab["###Nan###"]
        if self.nan_id in self.finite_ids or tokenizer.eos_token_id in self.finite_ids:
            raise ValueError("Missing/EOS token cannot be a finite output")
        self._finite_tensors = {}
        self.finite_set = frozenset(self.finite_ids)
        self.id_centers = dict(zip(self.finite_ids, np.unique(self.centers)))
        self._lookups = {}
        prompt = OFFICIAL_PROMPT.format("###0.0001###")
        pre = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        text = prompt+self.finite_words[0]+" "+self.finite_words[-1]
        full = tokenizer(text, add_special_tokens=True)["input_ids"]
        if full[:len(pre)] != pre:
            raise ValueError("Answer tokenization changes the prompt prefix")
        suffix = full[len(pre):]
        first, last = suffix.index(self.finite_ids[0]), suffix.index(self.finite_ids[-1])
        self.first_separator = tuple(suffix[:first])
        self.separator = tuple(suffix[first+1:last])
        if suffix[last+1:] or not self.separator:
            raise ValueError("Unexpected canonical numeric grammar")
        self.alphabet_sha = hashlib.sha256(json.dumps({"words": self.finite_words,
            "ids": self.finite_ids, "first": self.first_separator, "separator": self.separator},
            sort_keys=True).encode()).hexdigest()

    def alphabet(self, device):
        key = str(device)
        if key not in self._finite_tensors:
            self._finite_tensors[key] = torch.tensor(self.finite_ids, device=device, dtype=torch.long)
        return self._finite_tensors[key]

    def lookup(self, size, device):
        key = (size, str(device))
        if key not in self._lookups:
            table = torch.full((size,), -100, dtype=torch.long, device=device)
            table[self.alphabet(device)] = torch.arange(len(self.finite_ids), device=device)
            self._lookups[key] = table
        return self._lookups[key]

    def record(self, history, response=None):
        base = super().record(history)
        if response is None:
            return base
        if not base["present"]:
            raise ValueError("Training history must contain an observation")
        target = np.asarray(response, np.float64)
        if not 1 <= len(target) <= 24 or not np.isfinite(target).any():
            raise ValueError("One to 24 targets with at least one observed value required")
        # Missingness is metadata only. The decoder prefix never sees a future Nan.
        filled, previous = target.copy(), float(base["last"])
        for index, value in enumerate(filled):
            if np.isfinite(value):
                previous = float(value)
            else:
                filled[index] = previous
        scale, offset = base["scaling"]
        normalized = filled*scale+offset-.5
        bins = self.centers[np.digitize(normalized, self.boundaries, right=True)]
        answer = " ".join(f"###{x:.4f}###" for x in bins)
        prompt = self.serialize(history, np.isfinite(history), 0.)[0]
        ids = list(self.tokenizer(prompt+answer, add_special_tokens=True)["input_ids"])
        end = base["prompt_end"]+1
        if ids[:end] != base["ids"]:
            raise ValueError("Full answer changed prompt token coordinates")
        positions = [i for i in range(end, len(ids)) if ids[i] in self.finite_set]
        if len(positions) != len(target) or self.nan_id in ids[end:]:
            raise ValueError("Canonical answer must contain exactly H finite numeric tokens")
        expected = []
        for i, position in enumerate(positions):
            expected.extend(self.first_separator if i == 0 else self.separator)
            expected.append(ids[position])
        if ids[end:] != expected:
            raise ValueError("Whole-answer serialization and incremental grammar differ")
        labels = [-100]*len(ids)
        for position, observed in zip(positions, np.isfinite(target)):
            if observed:
                labels[position] = ids[position]
        base.update(ids=ids, labels=labels, answer_positions=positions,
                    target_observed=np.isfinite(target).tolist(), filled_response=filled.tolist(),
                    target_range_exceeded=((normalized < -.9999) | (normalized > .9999)).tolist())
        return base

    def decode_ids(self, ids, scale, offset, fallback):
        normalized = np.array([self.id_centers[int(i)] for i in ids], np.float64)
        raw = (normalized+.5-offset)/scale
        valid = np.isfinite(raw) & (np.abs(raw) <= np.finfo(np.float32).max)
        prediction = np.where(valid, raw, fallback).astype(np.float32)
        boundary = (normalized == -.9999) | (normalized == .9999)
        return prediction, valid, boundary

    def audit_alphabet(self):
        prompt = OFFICIAL_PROMPT.format("###0.0001###")
        pre = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        # Cache get_vocab once in __init__; never rebuild a 42001-word map per token.
        for word, token in zip(self.finite_words, self.finite_ids):
            full = self.tokenizer(prompt+word+" "+word, add_special_tokens=True)["input_ids"]
            expected = pre+list(self.first_separator)+[token]+list(self.separator)+[token]
            if full != expected:
                raise ValueError("Alphabet/context mismatch: "+word)
        return {"finite_tokens": len(self.finite_ids), "alphabet_sha256": self.alphabet_sha,
                "nan_excluded": True, "first_separator": self.first_separator,
                "separator": self.separator, "whole_and_incremental_checked": True}


def numeric_loss(logits, labels, codec):
    """logits at j-1 predict the observed numeric label at j; format is external."""
    shifted = labels[:, 1:]
    valid = shifted != -100
    counts = valid.sum(-1)
    if (counts == 0).any():
        raise ValueError("No observed numeric target in a training query")
    alphabet = codec.alphabet(logits.device)
    lookup = codec.lookup(logits.shape[-1], logits.device)
    compact = lookup[shifted.clamp_min(0)].masked_fill(~valid, -100)
    if ((compact == -100) & valid).any():
        raise ValueError("Nonfinite or nonnumeric supervised token")
    scores = logits[:, :-1].float().index_select(-1, alphabet)
    loss = F.cross_entropy(scores.transpose(1, 2), compact, ignore_index=-100, reduction="none")
    return (loss.sum(-1)/counts).sum(), int(counts.sum())
