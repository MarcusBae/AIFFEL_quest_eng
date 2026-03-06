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
from models import Transformer
from utils import (
    load_data, build_corpus, load_word2vec, augment_data, 
    tokenize_and_vectorize, pad_sequences, generate_masks, 
    loss_function, LearningRateScheduler, calculate_bleu, 
    visualize_attention, preprocess_sentence
)

warnings.filterwarnings(action='ignore')

def evaluate_with_attention(sentence, transformer, vocab, rev_vocab, device, max_len=40):
    transformer.eval()
    sentence = preprocess_sentence(sentence)
    pecab = PeCab()  # ideally this should be global or passed in, but moving it here is fine for now.
    src_tokens = pecab.morphs(sentence)

    enc_ids = [vocab.get(t, vocab["<unk>"]) for t in src_tokens]
    enc_in = torch.tensor([enc_ids], dtype=torch.long).to(device)
    
    # 1. 인코더 연산
    enc_mask = (enc_in == 0).unsqueeze(1).unsqueeze(2).float().to(device)
    enc_out, _ = transformer.encoder(transformer.embedding(transformer.enc_emb, enc_in), enc_mask)

    dec_in = torch.tensor([[vocab["<start>"]]], dtype=torch.long).to(device)
    last_cross_attn = None

    for _ in range(max_len):
        # 2. 디코더 연산
        dec_lookahead_mask = torch.triu(torch.ones(dec_in.shape[1], dec_in.shape[1]), diagonal=1).unsqueeze(0).unsqueeze(1).to(device)
        dec_tgt_padding_mask = (dec_in == 0).unsqueeze(1).unsqueeze(2).float().to(device)
        dec_mask = torch.max(dec_tgt_padding_mask, dec_lookahead_mask)
        
        # 인코더 결과(enc_out)에 대한 패딩 마스크는 enc_mask와 동일하게 사용
        dec_enc_mask = enc_mask 

        # 디코더만 호출 (Transformer.decoder 직접 호출)
        dec_out, _, dec_enc_attns = transformer.decoder(
            transformer.embedding(transformer.dec_emb, dec_in),
            enc_out,
            dec_enc_mask,
            dec_mask
        )
        logits = transformer.fc(dec_out)
        
        pred_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if dec_enc_attns is not None and len(dec_enc_attns) > 0:
            last_cross_attn = dec_enc_attns[-1]
            
        if pred_id.item() == vocab["<end>"]:
            break
        dec_in = torch.cat([dec_in, pred_id], dim=-1)

    out_ids = dec_in.squeeze(0).tolist()[1:]
    response_tokens = []
    for i in out_ids:
        if i == vocab["<end>"]:
            break
        response_tokens.append(rev_vocab.get(i, "<unk>"))

    return response_tokens, last_cross_attn, src_tokens

def main():
    # Settings
    DATA_PATH = 'data/ChatbotData.csv'
    WV_PATH = 'data/ko.bin'
    MAX_LEN = 40
    D_MODEL = 512
    N_HEADS = 8
    D_FF = 1024
    DROPOUT = 0.2
    N_LAYERS = 6
    BATCH_SIZE = 64
    EPOCHS = 1 # Set to a low number for quick verification if needed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Pipeline
    questions, answers = load_data(DATA_PATH)
    que_corpus, ans_corpus = build_corpus(questions, answers)
    wv = load_word2vec(WV_PATH)
    aug_que, aug_ans = augment_data(que_corpus, ans_corpus, wv)
    print(f"Total samples after augmentation: {len(aug_que)}")

    que_vector, ans_vector, vocab = tokenize_and_vectorize(aug_que, aug_ans)
    rev_vocab = {v: k for k, v in vocab.items()}

    enc_train = pad_sequences(que_vector, MAX_LEN)
    dec_train = pad_sequences(ans_vector, MAX_LEN)

    # Dataset / DataLoader
    dataset = TensorDataset(enc_train, dec_train)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Optimizer, Scheduler
    transformer = Transformer(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        src_vocab_size=len(vocab),
        tgt_vocab_size=len(vocab),
        pos_len=MAX_LEN,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LearningRateScheduler(D_MODEL)

    # Training Loop
    train_losses, val_losses = [], []
    total_step = 0

    for epoch in range(EPOCHS):
        transformer.train()
        total_train_loss = 0.0
        for b_enc, b_dec in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]"):
            b_enc, b_dec = b_enc.to(device), b_dec.to(device)
            dec_input, dec_real = b_dec[:, :-1], b_dec[:, 1:]
            enc_mask, dec_enc_mask, dec_mask = generate_masks(b_enc, dec_input, device)

            optimizer.param_groups[0]["lr"] = lr_scheduler(total_step)
            optimizer.zero_grad()
            pred = transformer(b_enc, dec_input, enc_mask, dec_enc_mask, dec_mask)
            loss = loss_function(dec_real, pred)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_step += 1
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        transformer.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for b_enc, b_dec in tqdm(val_loader, desc=f"Epoch {epoch+1} [val]"):
                b_enc, b_dec = b_enc.to(device), b_dec.to(device)
                dec_input, dec_real = b_dec[:, :-1], b_dec[:, 1:]
                enc_mask, dec_enc_mask, dec_mask = generate_masks(b_enc, dec_input, device)
                pred = transformer(b_enc, dec_input, enc_mask, dec_enc_mask, dec_mask)
                vloss = loss_function(dec_real, pred)
                total_val_loss += vloss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"[Epoch {epoch+1}] train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}")

    # Evaluation
    test_sentences = ["지루하다, 놀러가고 싶어.", "오늘 일찍 일어났더니 피곤하다.", "간만에 여자친구랑 데이트 하기로 했어.", "집에 있는다는 소리야."]
    qa_dict = {
        "지루하다, 놀러가고 싶어.": "놀러가면 기분이 전환될 거예요.",
        "오늘 일찍 일어났더니 피곤하다.": "잠깐 쉬어가는 것도 좋아요.",
        "간만에 여자친구랑 데이트 하기로 했어.": "즐거운 시간 보내세요.",
        "집에 있는다는 소리야.": "집에서 편히 쉬는 것도 좋죠."
    }

    print("\n# Evaluation Results")
    for ts in test_sentences:
        response, attention, tokens = evaluate_with_attention(ts, transformer, vocab, rev_vocab, device)
        print(f"\nQ: {ts}")
        print(f"A: {' '.join(response)}")
        if ts in qa_dict:
            ref_tokens = PeCab().morphs(qa_dict[ts])
            print(f"BLEU Score: {calculate_bleu(ref_tokens, response):.4f}")

if __name__ == "__main__":
    main()
