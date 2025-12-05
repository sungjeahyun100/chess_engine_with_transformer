from __future__ import annotations
from typing import List, Dict
import re


class CharTokenizer:
    def __init__(self, texts: List[str] | None = None):
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []
        if texts:
            self.fit(texts)

    def fit(self, texts: List[str]):
        vocab = sorted({ch for t in texts for ch in t})
        self.itos = ["<pad>", "<bos>", "<eos>"] + vocab
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        ids = [self.stoi.get(ch, 0) for ch in text]
        if add_special:
            return [1] + ids + [2]
        return ids

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos[i] for i in ids if i < len(self.itos))


_MOVE_RE = re.compile(r"\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?)\b")


class SimpleMoveTokenizer:
    """
    Heuristic move-level tokenizer extracting SAN-like tokens.
    Not a full PGN parser, but good enough for prototyping.
    """

    def __init__(self, games: List[str] | None = None):
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []
        if games:
            self.fit(games)

    def fit(self, games: List[str]):
        tokens = set()
        for g in games:
            tokens.update(self._extract_moves(g))
        base = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.itos = base + sorted(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def _extract_moves(self, pgn_text: str) -> List[str]:
        # Remove PGN headers like [Event "..."]
        lines = [ln for ln in pgn_text.splitlines() if not ln.strip().startswith('[')]
        text = ' '.join(lines)
        return _MOVE_RE.findall(text)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, pgn_text: str, add_special: bool = True) -> List[int]:
        toks = self._extract_moves(pgn_text)
        ids = [self.stoi.get(t, self.stoi.get("<unk>", 3)) for t in toks]
        if add_special:
            return [1] + ids + [2]
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = [self.itos[i] for i in ids if i < len(self.itos)]
        return ' '.join(toks)
