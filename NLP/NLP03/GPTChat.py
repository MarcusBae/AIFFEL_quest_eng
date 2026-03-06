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

# Import local modules
from models import GPTModel
from utils import (
    load_data, build_corpus, load_word2vec, augment_data, 
    tokenize_and_vectorize_gpt, pad_sequences, generate_gpt_masks, 
    loss_function, LearningRateScheduler, preprocess_sentence
)

warnings.filterwarnings(action='ignore')

def generate_text(model, start_tokens, vocab, rev_vocab, device, max_len=40):
    model.eval()
    input_ids = torch.tensor([start_tokens], dtype=torch.long).to(device)
    
    for _ in range(max_len):
        mask = generate_gpt_masks(input_ids, device)
        logits = model(input_ids, mask)
        pred_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        if pred_id.item() == vocab["<end>"]:
            break
        input_ids = torch.cat([input_ids, pred_id], dim=-1)
        
    # Extract only the generated part (after <sep>)
    full_ids = input_ids.squeeze(0).tolist()
    if vocab["<sep>"] in full_ids:
        sep_idx = full_ids.index(vocab["<sep>"])
        gen_ids = full_ids[sep_idx+1:]
    else:
        gen_ids = full_ids
        
    return [rev_vocab.get(i, "<unk>") for i in gen_ids]

def main():
    # Settings
    DATA_PATH = 'data/ChatbotData.csv'
    WV_PATH = 'data/ko.bin'
    MAX_LEN = 60 # GPT context length
    D_MODEL = 256
    N_HEADS = 8
    D_FF = 512
    DROPOUT = 0.1
    N_LAYERS = 2
    BATCH_SIZE = 64
    EPOCHS = 1 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Pipeline
    questions, answers = load_data(DATA_PATH)
    que_corpus, ans_corpus = build_corpus(questions, answers)
    
    # Optional augmentation
    wv = load_word2vec(WV_PATH)
    aug_que, aug_ans = augment_data(que_corpus, ans_corpus, wv)
    
    combined_vector, vocab = tokenize_and_vectorize_gpt(aug_que, aug_ans)
    rev_vocab = {v: k for k, v in vocab.items()}

    full_data = pad_sequences(combined_vector, MAX_LEN)

    # Dataset / DataLoader
    # Input is the sequence minus last token, Target is sequence shifted by 1
    dataset = TensorDataset(full_data)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Optimizer, Scheduler
    model = GPTModel(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        vocab_size=len(vocab),
        pos_len=MAX_LEN,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LearningRateScheduler(D_MODEL)

    # Training Loop
    total_step = 0
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        for (batch,) in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]"):
            batch = batch.to(device)
            # Input: <start> Q <sep> A
            # Target: Q <sep> A <end>
            x, y = batch[:, :-1], batch[:, 1:]
            mask = generate_gpt_masks(x, device)

            optimizer.param_groups[0]["lr"] = lr_scheduler(total_step)
            optimizer.zero_grad()
            pred = model(x, mask)
            loss = loss_function(y, pred)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_step += 1
        
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for (batch,) in tqdm(val_loader, desc=f"Epoch {epoch+1} [val]"):
                batch = batch.to(device)
                x, y = batch[:, :-1], batch[:, 1:]
                mask = generate_gpt_masks(x, device)
                pred = model(x, mask)
                vloss = loss_function(y, pred)
                total_val_loss += vloss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"[Epoch {epoch+1}] train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}")

    # Evaluation
    test_questions = ["지루하다, 놀러가고 싶어.", "오늘 일찍 일어났더니 피곤하다.", "너는 누구니?"]
    pecab = PeCab()
    print("\n# GPT Generation Results")
    for q in test_questions:
        q_clean = preprocess_sentence(q)
        tokens = pecab.morphs(q_clean)
        start_tokens = [vocab["<start>"]] + [vocab.get(t, vocab["<unk>"]) for t in tokens] + [vocab["<sep>"]]
        response = generate_text(model, start_tokens, vocab, rev_vocab, device)
        print(f"Q: {q}")
        print(f"A: {' '.join(response)}")

if __name__ == "__main__":
    main()
