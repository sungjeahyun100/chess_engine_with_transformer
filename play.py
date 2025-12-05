#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import torch
import chess #type: ignore
from ml.load_model import load_model_from_checkpoint
from ml.tokenizer import SimpleMoveTokenizer

def load_tokenizer_from_checkpoint(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'vocab' in ckpt:
        tokenizer = SimpleMoveTokenizer()
        tokenizer.itos = ckpt['vocab']
        tokenizer.stoi = {t: i for i, t in enumerate(tokenizer.itos)}
        return tokenizer
    raise ValueError("vocab not found in checkpoint")

def get_legal_moves_uci(board):
    """현재 보드에서 합법적인 수를 UCI 표기법으로 반환"""
    return [move.uci() for move in board.legal_moves]

def san_to_uci(board, san_move):
    """SAN 표기법을 UCI 표기법으로 변환"""
    try:
        return board.parse_san(san_move).uci()
    except:
        return None

def uci_to_san(board, uci_move):
    """UCI 표기법을 SAN 표기법으로 변환"""
    try:
        move = chess.Move.from_uci(uci_move)
        return board.san(move)
    except:
        return None

def sample_move(logits, tokenizer, temperature=1.0, top_k=20):
    logits = logits / temperature
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits[indices] = values
    probs = torch.softmax(logits, dim=-1)
    move_id = torch.multinomial(probs, 1).item()
    return move_id

def get_legal_move_from_model(model, tokenizer, device, sequence, board, temperature=1.0, top_k_sample=50):
    """모델이 생성한 수 중에서 합법적인 수를 찾기"""
    eos_idx = tokenizer.stoi.get("<eos>", 2)
    
    with torch.no_grad():
        input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
        logits = model(input_ids)
        last_logits = logits[0, -1, :]
        
        # 상위 k개 수를 가져오기
        probs = torch.softmax(last_logits, dim=-1)
        top_probs, top_indices = probs.topk(min(top_k_sample, len(tokenizer.itos)))
        
        legal_moves_uci = get_legal_moves_uci(board)
        
        # 상위 예측 중에서 합법적인 수 찾기
        for prob, idx in zip(top_probs, top_indices):
            move_token = tokenizer.itos[idx.item()] if idx.item() < len(tokenizer.itos) else None
            
            if move_token is None:
                continue
            
            # 토큰을 UCI로 변환 시도
            move_uci = None
            
            # 직접 UCI 형식일 수도 있음 (예: e2e4)
            if len(move_token) == 4 and move_token in legal_moves_uci:
                move_uci = move_token
            else:
                # SAN 형식일 수도 있음 (예: e4, Nf3)
                move_uci = san_to_uci(board, move_token)
            
            # 합법적인 수인지 확인
            if move_uci and move_uci in legal_moves_uci:
                return move_token, move_uci, prob.item(), top_probs, top_indices
    
    return None, None, 0.0, probs, []

def predict_next_moves(model, tokenizer, device, moves_str, top_k=5, temperature=1.0):
    """사용자가 입력한 수열에 대한 다음 수 예측"""
    print("\n" + "="*60)
    print("Next Move Prediction")
    print("="*60)
    
    bos_idx = tokenizer.stoi.get("<bos>", 1)
    
    moves = moves_str.split()
    sequence = [bos_idx]
    board = chess.Board()
    
    print("\nInput moves: %s" % ' '.join(moves))
    print("Board after moves:")
    print("-" * 60)
    
    # 입력된 수들을 시퀀스에 추가하고 보드 업데이트
    for move in moves:
        move_id = tokenizer.stoi.get(move, tokenizer.stoi.get("<unk>", 3))
        sequence.append(move_id)
        
        # 보드 업데이트
        move_uci = san_to_uci(board, move)
        if move_uci:
            board.push(chess.Move.from_uci(move_uci))
        else:
            print("Warning: Could not parse move %s" % move)
    
    print(board)
    print("\nFEN: %s" % board.fen())
    print("-" * 60)
    
    # 다음 수 예측
    with torch.no_grad():
        input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
        logits = model(input_ids)
        last_logits = logits[0, -1, :]
        
        probs = torch.softmax(last_logits, dim=-1)
        top_probs, top_indices = probs.topk(top_k)
        
        print("\nTop %d predictions (only legal moves):" % top_k)
        print("-" * 60)
        
        legal_moves_uci = get_legal_moves_uci(board)
        legal_count = 0
        
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            move_token = tokenizer.itos[idx.item()] if idx.item() < len(tokenizer.itos) else "?"
            
            # 합법성 확인
            move_uci = san_to_uci(board, move_token) if move_token != "?" else None
            is_legal = move_uci and move_uci in legal_moves_uci
            
            legal_marker = " [LEGAL]" if is_legal else " [ILLEGAL]"
            print("%d. %s: %.2f%%%s" % (i+1, move_token, 100*prob.item(), legal_marker))
            
            if is_legal:
                legal_count += 1
        
        print("-" * 60)
        print("Legal moves found in top %d: %d" % (top_k, legal_count))
    
    print("="*60 + "\n")

def play_game_interactive(model, tokenizer, device, temperature=1.0, player_color=None):
    """사용자와 상호작용하는 게임"""
    print("\n" + "="*60)
    print("Interactive Chess Game")
    print("="*60)
    
    bos_idx = tokenizer.stoi.get("<bos>", 1)
    sequence = [bos_idx]
    moves = []
    board = chess.Board()
    
    # 플레이어 색상 선택
    if player_color is None:
        print("\nChoose your color:")
        print("1. White (first move)")
        print("2. Black (second move)")
        color_choice = input("Select (1 or 2): ").strip()
        player_color = 'white' if color_choice == '1' else 'black'
    
    is_white_turn = True
    
    print("\nYou are playing as: %s" % player_color.upper())
    print("Type moves in PGN notation (e.g., 'e4', 'e5', 'Nf3')")
    print("Type 'board' to see current board")
    print("Type 'legal' to see legal moves")
    print("Type 'clear' to start over")
    print("Type 'quit' to exit")
    print("-" * 60)
    
    with torch.no_grad():
        while True:
            if board.is_game_over():
                print("\n*** Game Over ***")
                print("Result: %s" % board.result())
                break
            
            print("\nCurrent moves: %s" % ' '.join(moves) if moves else "Current moves: (none)")
            print("Board:")
            print(board)
            print()
            
            current_color = "White" if is_white_turn else "Black"
            is_player_turn = (player_color == 'white' and is_white_turn) or (player_color == 'black' and not is_white_turn)
            
            if is_player_turn:
                # 플레이어 턴
                while True:
                    user_input = input("Your move (%s): " % current_color).strip()
                    
                    if user_input.lower() == 'quit':
                        print("Exiting game")
                        return
                    elif user_input.lower() == 'board':
                        print(board)
                        continue
                    elif user_input.lower() == 'legal':
                        legal_moves = [board.san(move) for move in board.legal_moves]
                        print("Legal moves: %s" % ', '.join(legal_moves))
                        continue
                    elif user_input.lower() == 'clear':
                        sequence = [bos_idx]
                        moves = []
                        board = chess.Board()
                        is_white_turn = True
                        print("Cleared. Starting over.")
                        break
                    else:
                        # 사용자가 입력한 수를 검증
                        try:
                            move_uci = san_to_uci(board, user_input)
                            if not move_uci:
                                print("Invalid move: %s" % user_input)
                                continue
                            
                            move_obj = chess.Move.from_uci(move_uci)
                            if move_obj not in board.legal_moves:
                                print("Illegal move: %s" % user_input)
                                continue
                            
                            # 수를 진행
                            move_token = user_input
                            move_id = tokenizer.stoi.get(move_token, tokenizer.stoi.get("<unk>", 3))
                            
                            moves.append(move_token)
                            sequence.append(move_id)
                            board.push(move_obj)
                            
                            # 다음 수의 상위 예측 표시
                            input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
                            logits = model(input_ids)
                            last_logits = logits[0, -1, :]
                            
                            probs = torch.softmax(last_logits, dim=-1)
                            top_probs, top_indices = probs.topk(5)
                            
                            legal_moves_uci = get_legal_moves_uci(board)
                            top_moves_with_status = []
                            
                            for prob, idx in zip(top_probs, top_indices):
                                move_token_pred = tokenizer.itos[idx.item()] if idx.item() < len(tokenizer.itos) else "?"
                                move_uci_pred = san_to_uci(board, move_token_pred)
                                is_legal = move_uci_pred and move_uci_pred in legal_moves_uci
                                status = "✓" if is_legal else "✗"
                                top_moves_with_status.append("%s%s" % (move_token_pred, status))
                            
                            print("Top predictions: %s" % ' | '.join(top_moves_with_status))
                            is_white_turn = not is_white_turn
                            break
                        
                        except Exception as e:
                            print("Error: %s" % str(e))
                            continue
            
            else:
                # AI 턴
                print("AI's move (%s):" % current_color)
                
                # 합법적인 수 찾기
                move_token, move_uci, prob, top_probs, top_indices = get_legal_move_from_model(
                    model, tokenizer, device, sequence, board, temperature=temperature, top_k_sample=50)
                
                if not move_token or not move_uci:
                    print("ERROR: AI could not find a legal move!")
                    print("Available legal moves: %s" % ', '.join([board.san(m) for m in board.legal_moves]))
                    break
                
                move_obj = chess.Move.from_uci(move_uci)
                moves.append(move_token)
                sequence.append(tokenizer.stoi.get(move_token, tokenizer.stoi.get("<unk>", 3)))
                board.push(move_obj)
                
                print("AI move: %s (Prob: %.2f%%) | FEN: %s" % (move_token, 100*prob, board.fen()[:40]))
                
                is_white_turn = not is_white_turn

def play_game(model, tokenizer, device, max_moves=20, temperature=1.0):
    print("\n" + "="*60)
    print("Chess Model Output")
    print("="*60)
    
    bos_idx = tokenizer.stoi.get("<bos>", 1)
    sequence = [bos_idx]
    moves = []
    board = chess.Board()
    
    print("\nGenerating moves (max %d)" % max_moves)
    print("-" * 60)
    
    for step in range(max_moves):
        if board.is_game_over():
            print("\nGame over at step %d" % (step+1))
            print("Result: %s" % board.result())
            break
        
        # 합법적인 수 찾기
        move_token, move_uci, prob, top_probs, top_indices = get_legal_move_from_model(
            model, tokenizer, device, sequence, board, temperature=temperature, top_k_sample=50)
        
        if not move_token or not move_uci:
            print("Step %d: No legal move found!" % (step+1))
            break
        
        move_obj = chess.Move.from_uci(move_uci)
        moves.append(move_token)
        sequence.append(tokenizer.stoi.get(move_token, tokenizer.stoi.get("<unk>", 3)))
        board.push(move_obj)
        
        print("Step %d: %s | Prob: %.2f%% | FEN: %s" % (
            step+1, move_token, 100*prob, board.fen()[:50]))
    
    print("-" * 60)
    print("\nGenerated moves: %s" % ' '.join(moves))
    print("Total: %d moves" % len(moves))
    print("Final FEN: %s" % board.fen())
    print("="*60 + "\n")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: %s\n" % device)
    
    # Search both checkpoints and checkpoints_finetune directories
    available_checkpoints = []
    
    for dir_name in ["checkpoints", "checkpoints_finetune"]:
        ckpt_dir = Path(dir_name)
        if ckpt_dir.exists():
            for ckpt_path in sorted(ckpt_dir.rglob("*.pt")):
                available_checkpoints.append(ckpt_path)
                print("[%d] %s" % (len(available_checkpoints), ckpt_path))
    
    if not available_checkpoints:
        print("No checkpoints available")
        sys.exit(1)
    
    print()
    try:
        choice = int(input("Select checkpoint number: ")) - 1
        if choice < 0 or choice >= len(available_checkpoints):
            raise ValueError
        checkpoint_path = available_checkpoints[choice]
    except (ValueError, IndexError):
        print("Invalid selection")
        sys.exit(1)
    
    print("\nLoading model: %s" % checkpoint_path)
    try:
        model = load_model_from_checkpoint(checkpoint_path, device=device)
        tokenizer = load_tokenizer_from_checkpoint(checkpoint_path)
        print("Model loaded successfully")
        print("Tokenizer loaded (vocab size: %d)" % tokenizer.vocab_size)
    except Exception as e:
        print("Load failed: %s" % str(e))
        sys.exit(1)
    
    # 메뉴 선택
    while True:
        print("\n" + "="*60)
        print("MENU")
        print("="*60)
        print("1. Auto-generate moves")
        print("2. Interactive mode (input moves)")
        print("3. Predict next moves")
        print("4. Exit")
        print("="*60)
        
        menu_choice = input("Select option (1-4): ").strip()
        
        if menu_choice == '1':
            try:
                max_moves_input = input("\nMax moves (default: 20): ") or "20"
                temp_input = input("Temperature (default: 1.0): ") or "1.0"
                
                max_moves = int(max_moves_input)
                temp = float(temp_input)
                
                play_game(model, tokenizer, device, max_moves=max_moves, temperature=temp)
            except ValueError as e:
                print("Input error: %s" % str(e))
        
        elif menu_choice == '2':
            try:
                temp_input = input("\nTemperature (default: 1.0): ") or "1.0"
                temp = float(temp_input)
                play_game_interactive(model, tokenizer, device, temperature=temp)
            except ValueError as e:
                print("Input error: %s" % str(e))
        
        elif menu_choice == '3':
            moves_str = input("\nEnter moves (space-separated, e.g., 'e4 e5 Nf3'): ").strip()
            if moves_str:
                predict_next_moves(model, tokenizer, device, moves_str, top_k=5)
        
        elif menu_choice == '4':
            print("\nExit")
            break
        
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()