from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CharTokenizer:
    vocab: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(set(self.vocab)) != len(self.vocab):
            raise ValueError("vocab contains duplicate characters")
        self._stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
        self._itos: dict[int, str] = {i: ch for i, ch in enumerate(self.vocab)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(vocab=sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - set(self._stoi))
        if unknown:
            raise ValueError(f"unknown characters in text: {unknown!r}")
        return [self._stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos[i] for i in ids)
