import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def positional_encoding(pos, d_model):
    def cal_angle(position, i):
        return position / np.power(10000, (2*(i//2)) / np.float32(d_model))
    def get_posi_angle_vec(position):
        return [cal_angle(position, i) for i in range(d_model)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(pos)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return sinusoid_table

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads, self.d_model = num_heads, d_model
        self.depth = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        QK = torch.matmul(Q, K.transpose(-1, -2))
        scaled_qk = QK / math.sqrt(d_k)
        if mask is not None: scaled_qk += (mask * -1e9)
        attentions = F.softmax(scaled_qk, dim=-1)
        return torch.matmul(attentions, V), attentions

    def split_heads(self, x):
        bsz, seq_len, _ = x.size()
        return x.view(bsz, seq_len, self.num_heads, self.depth).permute(0, 2, 1, 3)

    def combine_heads(self, x):
        bsz, num_heads, seq_len, depth = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        WQ, WK, WV = self.W_q(Q), self.W_k(K), self.W_v(V)
        out, attn = self.scaled_dot_product_attention(self.split_heads(WQ), self.split_heads(WK), self.split_heads(WV), mask)
        return self.linear(self.combine_heads(out)), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.norm1, self.norm2 = nn.LayerNorm(d_model, eps=1e-6), nn.LayerNorm(d_model, eps=1e-6)
        self.do = nn.Dropout(dropout)
    def forward(self, x, mask):
        res = x
        out, attn = self.mha(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        out = self.do(out) + res
        res = out
        out = self.do(self.ffn(self.norm2(out))) + res
        return out, attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model, eps=1e-6), nn.LayerNorm(d_model, eps=1e-6), nn.LayerNorm(d_model, eps=1e-6)
        self.do = nn.Dropout(dropout)
    def forward(self, x, enc_out, dec_enc_mask, padding_mask):
        res = x
        out, attn1 = self.mha1(self.norm1(x), self.norm1(x), self.norm1(x), padding_mask)
        out = self.do(out) + res
        res = out
        out, attn2 = self.mha2(self.norm2(out), enc_out, enc_out, dec_enc_mask)
        out = self.do(out) + res
        res = out
        out = self.do(self.ffn(self.norm3(out))) + res
        return out, attn1, attn2

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
    def forward(self, x, mask):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attns.append(attn)
        return x, attns

class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
    def forward(self, x, enc_out, dec_enc_mask, padding_mask):
        attns, enc_attns = [], []
        for layer in self.layers:
            x, attn, enc_attn = layer(x, enc_out, dec_enc_mask, padding_mask)
            attns.append(attn)
            enc_attns.append(enc_attn)
        return x, attns, enc_attns

class Transformer(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, src_vocab_size, tgt_vocab_size, pos_len, dropout=0.2):
        super().__init__()
        self.d_model = float(d_model)
        self.enc_emb = nn.Embedding(src_vocab_size, d_model)
        self.dec_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.register_buffer("pos_encoding", torch.tensor(positional_encoding(pos_len, d_model), dtype=torch.float32))
        self.do = nn.Dropout(dropout)
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.fc = nn.Linear(d_model, tgt_vocab_size)
    def embedding(self, emb, x):
        seq_len = x.size(1)
        out = emb(x) * math.sqrt(self.d_model)
        out += self.pos_encoding[:seq_len, :].unsqueeze(0)
        return self.do(out)
    def forward(self, enc_in, dec_in, enc_mask, dec_enc_mask, dec_mask):
        enc_out, _ = self.encoder(self.embedding(self.enc_emb, enc_in), enc_mask)
        dec_out, _, _ = self.decoder(self.embedding(self.dec_emb, dec_in), enc_out, dec_enc_mask, dec_mask)
        return self.fc(dec_out)

class GPTLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.do = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self-Attention 블록
        res = x
        out, _ = self.mha(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        out = self.do(out) + res
        
        # Feed-Forward 블록
        res = out
        out = self.do(self.ffn(self.norm2(out))) + res
        return out

class GPTModel(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, vocab_size, pos_len, dropout=0.2):
        super().__init__()
        self.d_model = float(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_encoding", torch.tensor(positional_encoding(pos_len, d_model), dtype=torch.float32))
        self.do = nn.Dropout(dropout)
        
        # Transformer의 Encoder/Decoder 클래스와 완전히 독립된 레이어 스택
        self.layers = nn.ModuleList([GPTLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask):
        # x: (B, T)
        seq_len = x.size(1)
        out = self.embedding(x) * math.sqrt(self.d_model)
        out += self.pos_encoding[:seq_len, :].unsqueeze(0)
        out = self.do(out)

        # Causal Mask를 사용하는 GPT 전용 레이어들을 통과
        for layer in self.layers:
            out = layer(out, mask)
            
        return self.fc(out)
