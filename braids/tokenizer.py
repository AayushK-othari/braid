"""Tokenizer for BRAID strings, for use in chemical language models.

BRAID is designed to tokenize cleanly: every token is one of a small closed set
(atoms, `>k`, `^d`, bond symbols, `.`, stereo markers). This exposes the token
stream and a vocabulary builder so a CLM sees chemically-meaningful units rather
than raw characters.
"""
from __future__ import annotations
from .codec import _tokenize

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def tokenize(braid: str) -> list[str]:
    """Return the list of BRAID token strings (atoms, >k, ^d, bonds, ., %C/%T)."""
    return [text for _kind, text in _tokenize(braid)]


def token_kinds(braid: str) -> list[tuple[str, str]]:
    """Return (kind, text) pairs, e.g. ('organic','C'), ('branch','>2')."""
    return _tokenize(braid)


class Vocab:
    """Token<->id mapping built from a corpus of BRAID strings."""

    def __init__(self, tokens: list[str]):
        self.itos = list(SPECIAL) + [t for t in tokens if t not in SPECIAL]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def from_corpus(cls, braids: list[str]) -> "Vocab":
        seen = {}
        for b in braids:
            for t in tokenize(b):
                seen[t] = seen.get(t, 0) + 1
        # frequency-sorted for stable, compact ids
        ordered = sorted(seen, key=lambda t: (-seen[t], t))
        return cls(ordered)

    def __len__(self):
        return len(self.itos)

    def encode(self, braid: str, bos=True, eos=True) -> list[int]:
        ids = [self.stoi["<bos>"]] if bos else []
        unk = self.stoi["<unk>"]
        ids += [self.stoi.get(t, unk) for t in tokenize(braid)]
        if eos:
            ids.append(self.stoi["<eos>"])
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            t = self.itos[i] if 0 <= i < len(self.itos) else "<unk>"
            if t in SPECIAL:
                continue
            out.append(t)
        return "".join(out)
