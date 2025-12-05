from __future__ import annotations
from typing import List, Tuple, Sequence, Optional, Union, Dict
from pathlib import Path
import random
import re
import torch
from torch.utils.data import Dataset

try:
    import chess
    HAS_CHESS = True
except ImportError:
    HAS_CHESS = False


class PGNDataset(Dataset):
    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        tokenizer=None,
        max_seq_len: int = 512,
        split: str = "train",
        seed: int = 42,
        files: Optional[Sequence[Union[str, Path]]] = None,
        mask_loser: bool = False,
        penalize_illegal: bool = False,
    ):
        if files:
            all_files = [Path(f) for f in files]
            all_files = [f for f in all_files if f.exists() and f.is_file() and f.suffix.lower() == ".pgn"]
        else:
            if root is None:
                raise ValueError("Either root or files must be provided")
            root = Path(root)
            all_files = list(root.glob("*.pgn"))

        # Parse all games from all files
        self.games = []
        for file_path in all_files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                games_in_file = self._split_pgn_games(text)
                self.games.extend(games_in_file)
            except Exception:
                continue
        
        # Shuffle and split
        rng = random.Random(seed)
        rng.shuffle(self.games)
        n = len(self.games)
        cut = int(n * 0.9)
        if n == 1:
            cut = 1
        self.games = self.games[:cut] if split == "train" else self.games[cut:]
        
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.mask_loser = mask_loser
        self.penalize_illegal = penalize_illegal and HAS_CHESS
        
        if self.penalize_illegal and not HAS_CHESS:
            print("Warning: python-chess not installed. penalize_illegal disabled.")

        # Fit tokenizer if needed
        if getattr(tokenizer, "vocab_size", 0) == 0 and hasattr(tokenizer, "fit"):
            sample_games = self.games[:min(50, len(self.games))]
            tokenizer.fit(sample_games) #type: ignore
    
    def _split_pgn_games(self, text: str) -> List[Dict]:
        """Split PGN text into individual games with metadata"""
        games = []
        current_game = []
        
        for line in text.split('\n'):
            line = line.strip()
            if line:
                current_game.append(line)
            elif current_game:
                # End of game
                game_text = '\n'.join(current_game)
                if game_text:
                    game_data = self._extract_game_data(game_text)
                    if game_data:
                        games.append(game_data)
                current_game = []
        
        # Last game
        if current_game:
            game_text = '\n'.join(current_game)
            if game_text:
                game_data = self._extract_game_data(game_text)
                if game_data:
                    games.append(game_data)
        
        return games
    
    def _extract_game_data(self, pgn_text: str) -> Optional[Dict]:
        """Extract game moves and result"""
        # Extract result from PGN headers or end
        result_match = re.search(r'\[Result "([^"]+)"\]', pgn_text)
        result = result_match.group(1) if result_match else None
        
        # If no result in headers, check end of moves
        if not result:
            result_match = re.search(r'(1-0|0-1|1/2-1/2|\*)\s*$', pgn_text)
            result = result_match.group(1) if result_match else None
        
        # Extract white and black player names
        white_match = re.search(r'\[White "([^"]+)"\]', pgn_text)
        black_match = re.search(r'\[Black "([^"]+)"\]', pgn_text)
        white_player = white_match.group(1) if white_match else "Unknown"
        black_player = black_match.group(1) if black_match else "Unknown"
        
        return {
            'text': pgn_text,
            'result': result,
            'white': white_player,
            'black': black_player,
        }
    
    def _get_loser_color(self, result: Optional[str]) -> Optional[str]:
        """
        Get loser color from result string.
        Returns: 'white', 'black', or None (for draws)
        """
        if not result:
            return None
        
        if result == '1-0':  # White wins
            return 'black'
        elif result == '0-1':  # Black wins
            return 'white'
        elif result in ['1/2-1/2', '*']:  # Draw or unknown
            return None
        
        return None
    
    def _mask_loser_moves(self, pgn_text: str, loser_color: Optional[str], tokenizer) -> List[int]:
        """
        Encode PGN text with loser's moves masked.
        loser_color: 'white', 'black', or None
        """
        # Remove headers
        lines = [ln for ln in pgn_text.splitlines() if not ln.strip().startswith('[')]
        text_only = ' '.join(lines).strip()
        
        # Extract moves
        move_pattern = re.compile(r"\d+\.\s*")
        text_split = move_pattern.split(text_only)
        text_split = [t.strip() for t in text_split if t.strip()]
        
        ids = []
        ids.append(tokenizer.stoi.get("<bos>", 1))
        
        is_white_turn = True
        for part in text_split:
            moves_in_part = part.split()
            for move in moves_in_part:
                # Skip non-move tokens
                if move in ['*', '1-0', '0-1', '1/2-1/2']:
                    continue
                
                # Check if this move should be masked
                should_mask = False
                if loser_color == 'white' and is_white_turn:
                    should_mask = True
                elif loser_color == 'black' and not is_white_turn:
                    should_mask = True
                
                if should_mask:
                    # Use pad token for masked moves
                    ids.append(0)
                else:
                    move_id = tokenizer.stoi.get(move, tokenizer.stoi.get("<unk>", 3))
                    ids.append(move_id)
                
                is_white_turn = not is_white_turn
        
        ids.append(tokenizer.stoi.get("<eos>", 2))
        return ids

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        game_data = self.games[idx]
        text = game_data['text']
        
        if self.mask_loser and game_data['result']:
            loser_color = self._get_loser_color(game_data['result'])
            ids = self._mask_loser_moves(text, loser_color, self.tokenizer)
        else:
            ids = self.tokenizer.encode(text, add_special=True) #type: ignore
        
        # Truncate and create input/target for next-token prediction
        ids = ids[: self.max_seq_len + 1]
        if len(ids) < 2:
            ids = [1, 2]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        
        # 불법 수 페널티 마스크 생성 (optional)
        if self.penalize_illegal:
            penalty_mask = self._create_illegal_penalty_mask(text, ids)
            return x, y, penalty_mask
        
        return x, y
    
    def _create_illegal_penalty_mask(self, pgn_text: str, ids: List[int]) -> torch.Tensor:
        """
        불법 수에 페널티를 주기 위한 마스크 생성
        합법적인 수: 1.0, 불법 수: 0.0
        """
        board = chess.Board()
        penalty_mask = []
        
        is_white_turn = True
        
        for token_id in ids[1:]:  # BOS 제외
            # 토큰을 이동으로 변환
            if token_id == 0 or token_id >= len(self.tokenizer.itos):
                # PAD 또는 unknown
                penalty_mask.append(1.0)
                continue
            
            move_token = self.tokenizer.itos[token_id]
            
            # BOS, EOS 등 특수 토큰
            if move_token.startswith('<') and move_token.endswith('>'):
                penalty_mask.append(1.0)
                continue
            
            try:
                # SAN 형식의 수를 파싱
                move = board.parse_san(move_token)
                if move in board.legal_moves:
                    penalty_mask.append(1.0)  # 합법
                    board.push(move)
                    is_white_turn = not is_white_turn
                else:
                    penalty_mask.append(0.0)  # 불법
            except:
                penalty_mask.append(0.0)  # 파싱 실패 = 불법
        
        return torch.tensor(penalty_mask, dtype=torch.float)

    @staticmethod
    def collate(batch: List) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # penalize_illegal 여부에 따라 길이가 다름
        if len(batch[0]) == 3:
            xs, ys, penalty_masks = zip(*batch)
            max_len = max(x.size(0) for x in xs)
            pad_id = 0
            
            def pad(seq):
                if seq.size(0) < max_len:
                    return torch.cat([seq, torch.full((max_len - seq.size(0),), pad_id, dtype=seq.dtype)])
                return seq
            
            X = torch.stack([pad(x) for x in xs])
            Y = torch.stack([pad(y) for y in ys])
            attn_mask = (X != pad_id).long()
            
            # penalty mask 패딩 (불법 수로 취급)
            penalty_masks_padded = []
            for pm in penalty_masks:
                if pm.size(0) < max_len:
                    pm_padded = torch.cat([pm, torch.zeros(max_len - pm.size(0), dtype=pm.dtype)])
                else:
                    pm_padded = pm[:max_len]
                penalty_masks_padded.append(pm_padded)
            
            penalty_mask = torch.stack(penalty_masks_padded)
            
            return X, Y, attn_mask, penalty_mask
        else:
            xs, ys = zip(*batch)
            max_len = max(x.size(0) for x in xs)
            pad_id = 0
            
            def pad(seq):
                if seq.size(0) < max_len:
                    return torch.cat([seq, torch.full((max_len - seq.size(0),), pad_id, dtype=torch.long)])
                return seq
            
            X = torch.stack([pad(x) for x in xs])
            Y = torch.stack([pad(y) for y in ys])
            attn_mask = (X != pad_id).long()
            
            return X, Y, attn_mask

