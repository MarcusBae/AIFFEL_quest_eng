#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split
from pecab import PeCab
import warnings

# 프로젝트 내부 모듈 임포트
from models import GPTModel
from utils import (
    load_data, build_corpus, load_word2vec, augment_data, 
    tokenize_and_vectorize_gpt, tokenize_and_vectorize_pretrain,
    pad_sequences, generate_gpt_masks, 
    loss_function, LearningRateScheduler, preprocess_sentence
)

# 불필요한 경고 메시지 무시
warnings.filterwarnings(action='ignore')

def train_model(model, train_loader, val_loader, optimizer, lr_scheduler, device, epochs, stage_name, total_step_start=0):
    """
    모델 학습 및 검증을 수행하는 공통 함수
    """
    print(f"\n>>> Starting Stage: {stage_name}")
    total_step = total_step_start
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # 학습 단계
        model.train()
        current_train_loss = 0.0
        for (batch,) in tqdm(train_loader, desc=f"{stage_name} Epoch {epoch+1} [train]"):
            batch = batch.to(device)
            x, y = batch[:, :-1], batch[:, 1:]
            mask = generate_gpt_masks(x, device)

            optimizer.param_groups[0]["lr"] = lr_scheduler(total_step)
            optimizer.zero_grad()
            pred = model(x, mask)
            loss = loss_function(y, pred)
            loss.backward()
            optimizer.step()
            current_train_loss += loss.item()
            total_step += 1
        
        avg_train_loss = current_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 검증 단계
        model.eval()
        current_val_loss = 0.0
        with torch.no_grad():
            for (batch,) in tqdm(val_loader, desc=f"{stage_name} Epoch {epoch+1} [val]"):
                batch = batch.to(device)
                x, y = batch[:, :-1], batch[:, 1:]
                mask = generate_gpt_masks(x, device)
                pred = model(x, mask)
                vloss = loss_function(y, pred)
                current_val_loss += vloss.item()
        
        avg_val_loss = current_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"[{stage_name} Epoch {epoch+1}] train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}")

    return total_step

def generate_text(model, start_tokens, vocab, rev_vocab, device, max_len=40):
    """
    GPT 모델을 사용하여 텍스트를 생성하는 함수
    """
    model.eval()
    input_ids = torch.tensor([start_tokens], dtype=torch.long).to(device)
    
    for _ in range(max_len):
        mask = generate_gpt_masks(input_ids, device)
        logits = model(input_ids, mask)
        pred_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if pred_id.item() == vocab["<end>"]:
            break
        input_ids = torch.cat([input_ids, pred_id], dim=-1)
        
    full_ids = input_ids.squeeze(0).tolist()
    if vocab["<sep>"] in full_ids:
        sep_idx = full_ids.index(vocab["<sep>"])
        gen_ids = full_ids[sep_idx+1:]
    else:
        gen_ids = full_ids
        
    return [rev_vocab.get(i, "<unk>") for i in gen_ids]

def main():
    # 하이퍼파라미터 및 경로 설정
    DATA_PATH = 'data/ChatbotData.csv'
    WV_PATH = 'data/ko.bin'
    MODEL_PATH = 'pretrained_gpt.pth'
    MAX_LEN = 60
    D_MODEL = 256
    N_HEADS = 8
    D_FF = 512
    DROPOUT = 0.1
    N_LAYERS = 2
    BATCH_SIZE = 64
    PRETRAIN_EPOCHS = 1 
    FINETUNE_EPOCHS = 2 # 실습 편의상 낮은 값 설정

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 공통 데이터 로드 및 토큰화 준비
    questions, answers = load_data(DATA_PATH)
    que_corpus, ans_corpus = build_corpus(questions, answers)
    
    # 2. 어휘(Vocab) 생성 - 챗봇 포맷 기준 (<sep> 포함)
    wv = load_word2vec(WV_PATH)
    aug_que, aug_ans = augment_data(que_corpus, ans_corpus, wv)
    _, vocab = tokenize_and_vectorize_gpt(aug_que, aug_ans)
    rev_vocab = {v: k for k, v in vocab.items()}

    # 3. 모델 초기화
    model = GPTModel(
        n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
        vocab_size=len(vocab), pos_len=MAX_LEN, dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LearningRateScheduler(D_MODEL)

    # ==========================================
    # STAGE 1: Pre-training (Language Modeling)
    # ==========================================
    print("\n[STAGE 1] Pre-training on Question Corpus...")
    pretrain_vector = tokenize_and_vectorize_pretrain(que_corpus, vocab, MAX_LEN)
    pretrain_data = pad_sequences(pretrain_vector, MAX_LEN)
    
    dataset_pt = TensorDataset(pretrain_data)
    val_size_pt = int(len(dataset_pt) * 0.1)
    train_size_pt = len(dataset_pt) - val_size_pt
    train_ds_pt, val_ds_pt = random_split(dataset_pt, [train_size_pt, val_size_pt], generator=torch.Generator().manual_seed(42))
    
    train_loader_pt = DataLoader(train_ds_pt, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_pt = DataLoader(val_ds_pt, batch_size=BATCH_SIZE, shuffle=False)

    total_step = train_model(
        model, train_loader_pt, val_loader_pt, optimizer, 
        lr_scheduler, device, PRETRAIN_EPOCHS, "Pre-train"
    )
    
    # 가중치 저장
    print(f"Saving pre-trained weights to {MODEL_PATH}...")
    torch.save(model.state_dict(), MODEL_PATH)

    # ==========================================
    # STAGE 2: Fine-tuning (Chatbot Task)
    # ==========================================
    print("\n[STAGE 2] Fine-tuning on Q+A Corpus...")
    # 가중치 로드 (개념적 확인을 위해 다시 불러옴)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    combined_vector, _ = tokenize_and_vectorize_gpt(aug_que, aug_ans)
    finetune_data = pad_sequences(combined_vector, MAX_LEN)
    
    dataset_ft = TensorDataset(finetune_data)
    val_size_ft = int(len(dataset_ft) * 0.1)
    train_size_ft = len(dataset_ft) - val_size_ft
    train_ds_ft, val_ds_ft = random_split(dataset_ft, [train_size_ft, val_size_ft], generator=torch.Generator().manual_seed(42))
    
    train_loader_ft = DataLoader(train_ds_ft, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_ft = DataLoader(val_ds_ft, batch_size=BATCH_SIZE, shuffle=False)

    # total_step을 이어서 진행하여 LR 스케줄러 유지
    train_model(model, train_loader_ft, val_loader_ft, optimizer, lr_scheduler, device, FINETUNE_EPOCHS, "Fine-tune", total_step_start=total_step)

    # 7. 모델 평가 및 문장 생성 테스트
    test_questions = ["지루하다, 놀러가고 싶어.", "오늘 일찍 일어났더니 피곤하다.", "너는 누구니?"]
    pecab = PeCab()
    print("\n# Final GPT Generation Results")
    for q in test_questions:
        q_clean = preprocess_sentence(q)
        tokens = pecab.morphs(q_clean)
        start_tokens = [vocab["<start>"]] + [vocab.get(t, vocab["<unk>"]) for t in tokens] + [vocab["<sep>"]]
        response = generate_text(model, start_tokens, vocab, rev_vocab, device)
        print(f"Q: {q}")
        print(f"A: {' '.join(response)}")

if __name__ == "__main__":
    main()
