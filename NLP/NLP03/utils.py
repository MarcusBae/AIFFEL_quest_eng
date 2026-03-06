import os
import re
import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from pecab import PeCab
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import pickle
import gensim
import urllib.request

DATA_URL = 'https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv'
WV_URL = 'https://github.com/Kyubyong/wordvectors/raw/master/ko/ko.bin'

def load_data(path):
    if not os.path.exists(path):
        print(f"{path} not found. Downloading from {DATA_URL}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
        print("Download completed.")
    df = pd.read_csv(path)
    questions = df['Q'].tolist()
    answers = df['A'].tolist()
    return questions, answers

def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z가-힣0-9!?,.]", " ", sentence)
    sentence = re.sub(r" {2,}", " ", sentence)
    sentence = sentence.strip()
    return sentence

def build_corpus(questions, answers, max_len=40, corpus_path='data/corpus.pkl'):
    if os.path.exists(corpus_path):
        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, 'rb') as f:
            que_corpus, ans_corpus = pickle.load(f)
        return que_corpus, ans_corpus

    print("Building new corpus...")
    pecab = PeCab()
    que_corpus, ans_corpus = [], []
    seen = set()

    for q, a in tqdm(zip(questions, answers), total=len(questions), desc="Building Corpus"):
        q = preprocess_sentence(q)
        a = preprocess_sentence(a)
        if (q, a) in seen: continue
        seen.add((q, a))

        q_tokens = pecab.morphs(q)
        a_tokens = pecab.morphs(a)

        if len(q_tokens) <= max_len and len(a_tokens) <= max_len:
            que_corpus.append(q_tokens)
            ans_corpus.append(a_tokens)

    print(f"Saving corpus to {corpus_path}...")
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    with open(corpus_path, 'wb') as f:
        pickle.dump((que_corpus, ans_corpus), f)

    return que_corpus, ans_corpus

def load_word2vec(path):
    if not os.path.exists(path):
        print(f"{path} not found. Downloading from {WV_URL}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            urllib.request.urlretrieve(WV_URL, path)
            print("Download completed.")
        except Exception as e:
            print(f"Failed to download Word2Vec: {e}")
            return None

    try:
        model = gensim.models.Word2Vec.load(path)
        return model.wv
    except:
        try:
            return gensim.models.KeyedVectors.load(path)
        except:
            try:
                # Direct pickle load with latin1 (for legacy models)
                with open(path, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                if hasattr(data, 'syn0') and hasattr(data, 'index2word'):
                    from gensim.models.keyedvectors import KeyedVectors
                    kv = KeyedVectors(vector_size=data.vector_size if hasattr(data, 'vector_size') else data.syn0.shape[1])
                    kv.add_vectors(data.index2word, data.syn0)
                    return kv
                return None
            except:
                try:
                    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
                except:
                    return None

def lexical_sub(tokens, wv):
    if not tokens or wv is None: return tokens
    valid_tokens = [tok for tok in tokens if tok in wv]
    if not valid_tokens: return tokens
    selected_tok = random.choice(valid_tokens)
    similar_word = wv.most_similar(selected_tok)[0][0]
    return [similar_word if tok == selected_tok else tok for tok in tokens]

def augment_data(que_corpus, ans_corpus, wv):
    augmented_que, augmented_ans = [], []
    if wv is None:
        print("Word2Vec model not found. Skipping augmentation.")
        return que_corpus, ans_corpus
    for q, a in tqdm(zip(que_corpus, ans_corpus), total=len(que_corpus), desc="Augmenting"):
        augmented_que.append(q)
        augmented_ans.append(a)
        augmented_que.append(lexical_sub(q, wv))
        augmented_ans.append(a)
        augmented_que.append(q)
        augmented_ans.append(lexical_sub(a, wv))
    return augmented_que, augmented_ans

def tokenize_and_vectorize(que_corpus, ans_corpus):
    vocab = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3}
    for sentence in que_corpus + ans_corpus:
        for word in sentence:
            if word not in vocab: vocab[word] = len(vocab)
    def sentence_to_ids(sentence, vocab, is_target=False):
        ids = [vocab["<start>"]] if is_target else []
        ids.extend([vocab.get(word, vocab["<unk>"]) for word in sentence])
        if is_target: ids.append(vocab["<end>"])
        return ids
    que_vector = [sentence_to_ids(s, vocab) for s in que_corpus]
    ans_vector = [sentence_to_ids(s, vocab, is_target=True) for s in ans_corpus]
    return que_vector, ans_vector, vocab

def pad_sequences(sequences, max_len, pad_value=0):
    padded = np.full((len(sequences), max_len), pad_value)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    return torch.tensor(padded, dtype=torch.long)

def generate_padding_mask(seq):
    return (seq == 0).unsqueeze(1).unsqueeze(2).float()

def generate_lookahead_mask(size):
    return torch.triu(torch.ones(size, size), diagonal=1)

def generate_masks(src, tgt, device):
    enc_mask = generate_padding_mask(src).to(device)
    dec_enc_mask = generate_padding_mask(src).to(device)
    dec_lookahead_mask = generate_lookahead_mask(tgt.shape[1]).unsqueeze(0).unsqueeze(1).to(device)
    dec_tgt_padding_mask = generate_padding_mask(tgt).to(device)
    dec_mask = torch.max(dec_tgt_padding_mask, dec_lookahead_mask)
    return enc_mask, dec_enc_mask, dec_mask

def loss_function(real, pred):
    loss_ = F.cross_entropy(pred.contiguous().view(-1, pred.size(-1)), real.contiguous().view(-1), reduction='none')
    mask = (real != 0).float()
    return (loss_.view(real.size()) * mask).sum() / mask.sum()

class LearningRateScheduler:
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model, self.warmup_steps = d_model, warmup_steps
    def __call__(self, step):
        step = float(step + 1)
        return (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))

def calculate_bleu(reference, candidate):
    return sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)

def visualize_attention(sentence, response_tokens, attention, src_tokens):
    if attention is None:
        print("attention이 None이라 시각화할 수 없음")
        return
    if len(attention.shape) == 4:
        attention = attention.squeeze(0)
    attn_map = attention.mean(dim=0).detach().cpu().numpy()
    T_dec = len(response_tokens)
    T_enc = len(src_tokens)
    attn_map = attn_map[:T_dec, :T_enc]

    fig = plt.figure(figsize=(max(4, T_enc * 0.35), max(3, T_dec * 0.35)))
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(attn_map, cmap='viridis')
    fig.colorbar(cax)
    ax.set_xticks(range(T_enc))
    ax.set_yticks(range(T_dec))
    ax.set_xticklabels(src_tokens, rotation=90)
    ax.set_yticklabels(response_tokens)
    ax.set_xlabel("Input tokens (Encoder)")
    ax.set_ylabel("Output tokens (Decoder)")
    ax.set_title("Cross-Attention Map")
    plt.tight_layout()
    plt.show()

def tokenize_and_vectorize_gpt(que_corpus, ans_corpus, max_len=80):
    # GPT format: <start> Q <sep> A <end>
    vocab = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3, "<sep>": 4}
    for sentence in que_corpus + ans_corpus:
        for word in sentence:
            if word not in vocab: vocab[word] = len(vocab)
    
    combined_vectors = []
    for q, a in zip(que_corpus, ans_corpus):
        ids = [vocab["<start>"]]
        ids.extend([vocab.get(w, vocab["<unk>"]) for w in q])
        ids.append(vocab["<sep>"])
        ids.extend([vocab.get(w, vocab["<unk>"]) for w in a])
        ids.append(vocab["<end>"])
        combined_vectors.append(ids)
        
    return combined_vectors, vocab

def generate_gpt_masks(seq, device):
    lookahead_mask = torch.triu(torch.ones(seq.shape[1], seq.shape[1]), diagonal=1).unsqueeze(0).unsqueeze(1).to(device)
    padding_mask = (seq == 0).unsqueeze(1).unsqueeze(2).float().to(device)
    return torch.max(padding_mask, lookahead_mask)
