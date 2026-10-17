"""Stateless official ChatTime prompts and explicit time-to-token coordinates."""
from dataclasses import dataclass
import hashlib
import re

import numpy as np
import torch

from gofa_f.chattime import HistoryCodec
from gofa_f.generative_model import GenerativeCodec


OFFICIAL_PROMPT = ("You are a helpful assistant that performs time series prediction. "
    "The user will provide a sequence and you will predict the sequence.\n"
    "### Instruction:\nPlease predict the following sequence carefully.\n\n"
    "### Input:\n{}\n### Response:\n")
PROMPT_SHA = hashlib.sha256(OFFICIAL_PROMPT.encode()).hexdigest()


@dataclass
class AllNodeTokens:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_ends: torch.Tensor
    value_positions: tuple
    prompt_hashes: tuple
    full_ids: torch.Tensor

    def to(self, device):
        return AllNodeTokens(self.input_ids.to(device), self.attention_mask.to(device),
            self.labels.to(device), self.prompt_ends.to(device),
            tuple(p.to(device) for p in self.value_positions), self.prompt_hashes,
            self.full_ids.to(device))

    def append(self, ids):
        ids = ids.reshape(len(self.input_ids), -1)
        full = torch.cat((self.full_ids, ids), 1)
        mask = torch.cat((self.attention_mask, torch.ones_like(ids, dtype=torch.bool)), 1)
        return AllNodeTokens(ids, mask, torch.full_like(ids, -100), self.prompt_ends,
                             self.value_positions, self.prompt_hashes, full)

    def first(self, length):
        return AllNodeTokens(self.input_ids[:, :length], self.attention_mask[:, :length],
            self.labels[:, :length], self.prompt_ends, self.value_positions,
            self.prompt_hashes, self.full_ids[:, :length])


class AllNodeCodec(GenerativeCodec):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        numeric = re.compile(r"^###(?:Nan|[-+]?\d+\.\d+)###$")
        self.numeric_ids = {value for word, value in tokenizer.get_vocab().items() if numeric.match(word)}
        if not self.numeric_ids:
            raise ValueError("ChatTime numeric vocabulary missing")

    def serialize(self, values, observed, fallback=0.):
        # All scaling variables are call-local; no fitted sklearn object is shared.
        old, last, safe, present, scale, offset = HistoryCodec.serialize(self, values, observed, fallback)
        words = old.split("### Input:\n", 1)[1].rsplit("\n\n### Response:\n", 1)[0]
        return OFFICIAL_PROMPT.format(words), last, safe, present, scale, offset

    def record(self, history, response=None):
        result = super().record(history, response)
        positions = [i for i, token in enumerate(result["ids"][:result["prompt_end"]+1])
                     if token in self.numeric_ids]
        if len(positions) != len(history):
            raise ValueError("Exactly one numeric value token per historical time index required")
        result["value_positions"] = positions
        result["prompt_hash"] = hashlib.sha256(np.asarray(
            result["ids"][:result["prompt_end"]+1], np.int64).tobytes()).hexdigest()
        # Immutable scalars belong to this node/rolling block only.
        result["scaling"] = (float(result["inverse_scale"]), float(result["inverse_offset"]))
        return result

    def batch(self, records):
        tokens = super().batch(records)
        return AllNodeTokens(tokens.input_ids, tokens.attention_mask, tokens.labels, tokens.prompt_ends,
            tuple(torch.tensor(r["value_positions"], dtype=torch.long) for r in records),
            tuple(r["prompt_hash"] for r in records), tokens.input_ids.clone())

    def analysis_inputs(self, history, question):
        history = np.asarray(history, np.float64)
        prediction_prompt = self.serialize(history, np.isfinite(history), 0.)[0]
        words = prediction_prompt.split("### Input:\n", 1)[1].rsplit("\n### Response:\n", 1)[0]
        prompt = ("You are a helpful assistant that performs time series analysis. The user will provide "
            "a sequence and you will respond to the questions based on this sequence.\n"
            "### Instruction:\nPlease answer the following question carefully after analyzing the sequence: "
            +str(question)+"\n\n### Input:\n"+words+"\n### Response:\n")
        return self.tokenizer(prompt, return_tensors="pt")

    @staticmethod
    def parse_with_reasons(text, scale, offset, fallback, horizon):
        parts = re.findall(r"###(.*?)###", text)
        values = np.full(horizon, fallback, np.float32)
        valid = np.zeros(horizon, bool)
        reasons = []
        for i in range(horizon):
            reason = "missing"
            if i < len(parts):
                try:
                    value = (float(parts[i])+.5-offset)/scale
                    if np.isfinite(value) and abs(value) <= np.finfo(np.float32).max:
                        values[i], valid[i], reason = value, True, "valid"
                    else:
                        reason = "nonfinite_or_overflow"
                except (ValueError, OverflowError, ZeroDivisionError):
                    reason = "invalid"
            reasons.append(reason)
        return values, valid, reasons

