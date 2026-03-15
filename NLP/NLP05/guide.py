# -----------------------------------------------------------------------------
# 18.LLM Trend Note 2 [프로젝트]
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 알아야 할 것들
#   ChatGPT 구현을 위해 필요한 데이터셋의 종류 및 특징
#   Initial Model, Reward Model, RLHF Model의 학습 로직
#   KoChatGPT를 개선해 나만의 ChatGPT를 구현

#   Supervised Fine Tuning
#   Reward Model의 ranking algorithm 및 loss fuction 설계 원리
#   언어모델을 강화학습하기 위한 방법론

# -----------------------------------------------------------------------------
# 1. 준비
# # $ git clone https://github.com/airobotlab/KoChatGPT
# $ cp -r ./KoChatGPT/colossalai_ChatGPT_230319/chatgpt ./chatgpt
# $ pip install lm-eval loralib

import os
import json
import logging
import copy
from copy import deepcopy
import random
import functools
from typing import Optional, Dict, Sequence, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    PreTrainedTokenizerFast,
    GPT2Config,
    GPT2Model,
    pipeline
)

from chatgpt.dataset import RewardDataset
from chatgpt.models.base import RewardModel
from chatgpt.trainer.strategies import NaiveStrategy
from chatgpt.trainer.rm import RewardModelTrainer
from chatgpt.models.gpt import GPTActor, GPTCritic
from chatgpt.trainer import PPOTrainer

# -----------------------------------------------------------------------------
# Global Constants & Configuration
# -----------------------------------------------------------------------------

PROMPT_TEMPLATE = ">>> Instruction:\n{prompt}\n>>> Response:"

DEFAULT_TEST_PROMPTS = [
    '불고기용 고기 한우에요?',
    '리처드 닉슨이 43대 부통령직을 수행한 년도는?',
    '시카고 오헤어 국제공항은 어디에 있어?',
    '오늘 미세먼지 어때?'
]

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def clear_device_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.empty_cache()

def load_jsonl(path):
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        return []
    with open(path, "r", encoding='utf-8-sig') as f:
        return json.load(f)


def prepare_pairwise_data(list_data_dict):
    """
    kochatgpt_2_RM.jsonl -> (prompt, chosen, rejected) 쌍 생성
    """
    total_pairs = []
    for item in list_data_dict:
        prompt = item['prompt']
        ranking = item['ranking']
        completions = [item['completion_0'], item['completion_1'], item['completion_2']]
        
        # 가능한 모든 쌍 (0,1), (0,2), (1,2)에 대해 비교
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            if ranking[i] < ranking[j]:
                total_pairs.append({'prompt': prompt, 'chosen': completions[i], 'rejected': completions[j]})
            else:
                total_pairs.append({'prompt': prompt, 'chosen': completions[j], 'rejected': completions[i]})
    return total_pairs

def print_func_name(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n" + "-"*20 + f" {func.__name__}() " + "-"*20)
        return func(*args, **kwargs)
    return wrapper

def run_evaluation(cfg):
    model_paths = [cfg["sft_saved_dir"], cfg["ppo_saved_dir"]]
    common_args = {
         #"tokenizer": "skt/kogpt2-base-v2",
         "tokenizer": cfg["sft_tokenizer"],
        "dtype": "float16",
        "tasks": "kobest_copa,kobest_hellaswag,kobest_boolq",
        "batch_size": "auto",
        "limit": 500
    }

    for path in model_paths:
        script = (
            f"lm-eval --model hf "
            f"--model_args pretrained={path},tokenizer={common_args['tokenizer']},dtype={common_args['dtype']} "
            f"--tasks {common_args['tasks']} "
            f"--batch_size={common_args['batch_size']} "
            f"--limit={common_args['limit']}"
        )
        print(f"🚀 Running evaluation for: {path}")
        os.system(script)

# -----------------------------------------------------------------------------
# Pipeline API - high level
@print_func_name
def generate_pipeline(prompts, model_path, tokenizer):
    """
    리스트 전체를 파이프라인에 던져 처리, Beam Search와 Penalty 덕분에 문장이 꼬이거나 반복되는 현상이 훨씬 적음

    """
    # 파이프라인 생성
    generator = transformers.pipeline(
        'text-generation',
        model=model_path,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1 # GPU 자동 할당
    )

    # 생성 옵션 설정
    generation_args = dict(
        num_beams=4,            # 빔 서치의 너비. 클수록 더 좋은 문장을 생성할 확률이 높지만 속도가 느려짐.
        repetition_penalty=2.0, # 반복 패널티. 1.0보다 크면 반복을 억제함.
        no_repeat_ngram_size=4, # 값을 2~3으로 조정하면 단어 뭉치가 반복되는 것을 더 세밀하게 막을 수 있다.
        eos_token_id=tokenizer.eos_token_id,    # 문장 생성 종료 토큰
        max_new_tokens=64,      # 생성할 최대 토큰 수
        do_sample=True,         # sampling을 통해 더 다양한 문장을 생성한다.
        top_k=50,               # 값을 40~50 정도로 설정하면 모델이 다음 단어를 선택할 때 확률이 높은 상위 50개 단어 중에서만 고르게 된다. 
                                # (낮을수록 일관성↑, 높을수록 다양성↑)
        early_stopping=True     # 조기 종료
    )

    list_result = generator(prompts, **generation_args)
    
    results = [result[0]['generated_text'] for result in list_result]        
    return results

# Manual implementation - low level
@print_func_name
def generate_custom(prompts, model, tokenizer, device='cpu'):
    """
    하나씩 루프를 돌며 생성, Top-P Sampling을 통해 매번 조금씩 다른 창의적인 답변을 생성
    model.generate를 직접 호출 (Sampling 중심)
    """
    model.to(device)
    results = []
    
    for input_text in prompts:
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        outputs = model.generate(
            input_ids,
            max_length=250,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id # 패딩 토큰 설정 권장
        )
        
        # 디코딩 (outputs[0]는 전체 시퀀스이므로 바로 decode)
        # output_text = tokenizer.decode(outputs[0], skip_special_tokens=True) 
        output_text = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]
        results.append(output_text)
        
    return results

@print_func_name
def show_info(cfg):
    print(f"Torch version: {torch.__version__}") # Torch version:1.12.1
    print(f"transformers version: {transformers.__version__}") # transformers 4.28.0
    
    for k, v in cfg.items():
        print(f"{k}: {v}")

# -----------------------------------------------------------------------------
# 2. Base model and Dataset for RLHF
@print_func_name
def show_base_model_and_dataset(cfg, model, tokenizer):

    # 베이스라인 모델일반적인 성능을 확인하기
    # print('--- 1 ------------------------------------------------')
    r = tokenizer.model_max_length  # 입력 최대 토큰 수
    r = model.config.n_positions    # 모델이 입력받아 처리할 수 있는 최대 토큰 수 ?

    input_txt = "바람도 없는 공중에 수직의 파문을 내이며 고요히 떨어지는 오동잎은 누구의 발자취 입니까."
    tokens = tokenizer(input_txt).tokens()
    input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].numpy()

    pd.options.display.max_columns = 40
    pd.options.display.max_rows = 60

    df = pd.DataFrame([tokens, input_ids[0]], index=["kogpt-2_tokens", "Input_IDs"])
    print('--- 2 ------------------------------------------------')
    print(df)

    # 디코딩 성능 확인 -> 시퀀스 반복 출력이 왜 그리디 서치의 전형적인 현상인가?
    max_length = 128
    input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(cfg["device"])
    output_greedy = model.generate(input_ids, max_length=max_length, do_sample=False)
    print('--- 3 ------------------------------------------------')
    print(tokenizer.decode(output_greedy[0]))

    # 빔 서치 디코딩 사용, n-gram 패널티까지 부과 -> 뭔가 말이 되도록 나오는것 같다.
    print('--- 4 ------------------------------------------------')
    input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(cfg["device"])
    output_beam = model.generate(input_ids, max_length=max_length, do_sample=False,
                            num_beams=10, no_repeat_ngram_size=2)
    print(tokenizer.decode(output_beam[0]))

    # 샘플링 기법 추가
    # temperature : temperature 값을 낮추면(예: temperature=0.5) 확률이 높은 단어들이 더욱 우세하게 선택되며, 보수적인 생성
    # 반대로 값을 높이면(예: temperature=1.5) 확률이 낮은 단어들도 더 자주 선택, 창의적 + 불안정한 결과

    # top_k: 샘플링 시 상위 k개의 단어 후보만 고려하여 나머지를 무시
    # top_k=50 -> 확률이 높은 상위 50개의 단어만 샘플링 대상
    print('--- 5 ------------------------------------------------')
    output_beam = model.generate(input_ids, max_length=max_length, do_sample=True,
                            num_beams=7, no_repeat_ngram_size=2,
                            temperature=2.0, top_k=50)
    print(tokenizer.decode(output_beam[0]))
    # print(tokenizer.decode(output_beam[1])) # out of bound

    # top_p - Nucleus Sampling, 단어 선택 시 확률 누적 기준으로 상위 p%의 단어들만 고려
    print('--- 6 ------------------------------------------------')
    output_beam = model.generate(input_ids, max_length=max_length, do_sample=True,
                            num_beams=7, no_repeat_ngram_size=2,
                            top_p=0.90)
    print(tokenizer.decode(output_beam[0]))

    # todo
    # 1. 최선의 디코딩 방법을 위한 실험값 찾기
    #   - 빔사이즈, n-gram 패널티, temperature와 샘플링 인자로 조합
    #   - 구체적인 instruction과 prompting을 사용해 어떻게 디코딩을 해내는지도 확인해보세요.
    #   - RLHF를 적용하기 전의 실험값을 정리해보면 KoChatGPT의 성능 개선 여부를 확인하는데 도움이 될 것입니다.
    # kogpt-2 : 오리지널 GPT2의 가장 작은 버전

# SFT Supervised Fine Tuning 시도할 initial 모델 준비
@print_func_name
def show_sft_and_rm_dataset(cfg):

    # SFT DataSet - prompt, completion, tokens
    list_data_dict = load_jsonl(cfg['data_path_1_SFT'])

    r = f'{len(list_data_dict):,}'
    print(r)
    r = list_data_dict[:3]
    print(r)

    # RM DataSet - prompt, completion_0, 1, 2, ranking
    list_data_dict = load_jsonl(cfg['data_path_2_RM'])

    r = f'{len(list_data_dict):,}'
    print(r)
    r = list_data_dict[:3]
    print(r)

    # PPO (Proximal Policy Optimization) 학습에 쓰일 데이터 준비 - prompt only
    list_data_dict = load_jsonl(cfg['data_path_3_PPO'])

    r = f'{len(list_data_dict):,}'
    print(r)
    r = list_data_dict[:3]
    print(r)

# -----------------------------------------------------------------------------
# 3. Supervised Fine-Tuning
# SFT : kogpt-2를 instruction dataset으로 SFT 진행

# 모델 인퍼런스 단계에서 사용할 prompt 딕셔너리 템플릿과 SFT 데이터셋 클래스를 정의
class SFT_dataset(Dataset):
    def __init__(self, data_path_1_SFT: str, tokenizer: transformers.PreTrainedTokenizer, verbose=False):
        super(SFT_dataset, self).__init__()
        logging.info("SFT_dataset::__init__ - Loading data...")

        pattern_instruction = 'prompt'  # instruction
        pattern_output = 'completion'  # response

        list_data_dict = load_jsonl(data_path_1_SFT)

        prompt_input = PROMPT_TEMPLATE

        sources = []
        for example in list_data_dict:
            tmp = prompt_input.format_map(example)
            sources.append(tmp)

        targets = []
        for example in list_data_dict:
            targets.append(f"{example[pattern_output]}{tokenizer.eos_token}")
        examples = [s + t for s, t in zip(sources, targets)]

        sources_tokenized = self._tokenize_fn(sources, tokenizer)  # source
        examples_tokenized = self._tokenize_fn(examples, tokenizer)  # source + target

        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = -100   # ?? 무슨 의미인가? ignore_index ?

        data_dict = dict(input_ids=input_ids, labels=labels)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        logging.info(f"SFT_dataset::__init__ - Loading data done!!: {len(self.labels):,} samples")

    def _tokenize_fn(self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids, labels=labels, input_ids_lens=input_ids_lens, labels_lens=labels_lens,
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


# 하나의 학습용 배치(Batch) 생성. 
#   - Padding(패딩) 작업 
@dataclass
class DataCollatorForSupervisedDataset(object):
    # type hinting, @dataclass : parameter를 받는 생성자 대용 역할
    tokenizer: transformers.PreTrainedTokenizer

    # Sequence : 순서가 있는 시퀀스 - list, tuple
    # instances : batch_size 만큼의 데이터, SFT_dataset에서 반환된 데이터
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        # 이미 토크나이징 단계에서 input_ids에 padding_value가 포함되어 있다. 
        # 하지만, torch.nn.utils.rnn.pad_sequence의 역할은 배치 내의 시퀀스들을 
        # 가장 긴 시퀀스 길이에 맞춰 패딩하는 것이다. ???
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # padding_value = -100 : 학습 시 padding 은 Loss 계산에서 제외
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value= -100)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id), # 어디까지가 실제 데이터(1)이고 어디가 패딩(0)인지 표시
        )

# SFT(Supervised Fine-Tuning) 수행 후 결과 확인
@print_func_name
def run_sft(cfg, model, tokenizer):

    # 학습용 배치 Batch 생성기
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    train_dataset = SFT_dataset(
        data_path_1_SFT=cfg['data_path_1_SFT'],
        tokenizer=tokenizer
    )

    training_args = transformers.TrainingArguments(
        output_dir=cfg['sft_output_dir'],
        num_train_epochs=cfg['sft_num_train_epochs'],
        per_device_train_batch_size=cfg['sft_per_device_train_batch_size'],
        per_device_eval_batch_size=cfg['sft_per_device_eval_batch_size'],
        warmup_steps=cfg['sft_warmup_steps'],
        prediction_loss_only=cfg['sft_prediction_loss_only'],   
        fp16 = cfg['sft_fp16']
    )

    # Trainer : 학습 파이프라인 자동화 (Loop, Optimization, Logging)
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    if not os.path.exists(os.path.join(cfg['sft_saved_dir'], 'config.json')):
        if not os.path.exists(cfg['sft_saved_dir']):
            os.makedirs(cfg['sft_saved_dir'])
        trainer.train() # 학습 시작
        model.save_pretrained(cfg['sft_saved_dir'])
    else:
        print(f">>> SFT model already exists: {cfg['sft_saved_dir']}")

# -----------------------------------------------------------------------------
# 4. Reward Model
# -----------------------------------------------------------------------------

# GPTRM_custom: GPT 모델 뒤에 Reward를 출력하기 위한 Scalar Head(Linear layer)가 붙은 커스텀 모델.
# 사전 학습된 GPT-2 모델 + Scalar Head(Linear layer) 생성
# GPT-2는 다음 단어를 예측하는 모델이지만, 여기서는 마지막에 1개의 숫자(스칼라)만 출력하도록 개조되어 "이 문장의 점수는 몇 점이다를 출력
class GPTRM_custom(RewardModel):
    def __init__(self,
                pretrained: str | None = None,
                config: GPT2Config | None = None,
                checkpoint: bool = False,
                lora_rank: int = 0,
                lora_train_bias: str = 'none',
                tokenizer=None) -> None:

        if pretrained is not None:
            model = GPT2Model.from_pretrained(pretrained) # Class-Data, Architecture-Weights
            model.resize_token_embeddings(len(tokenizer))
        elif config is not None:
            model = GPT2Model(config)
        else:
            model = GPT2Model(GPT2Config())
        if checkpoint:
            model.gradient_checkpointing_enable()

        # 중요 - value_head : 입력된 문장의 점수를 계산하는 부분
        # model.config.n_embd: GPT-2의 내부 은닉 상태 벡터의 차원(예: 768, 1024 등).
        # 1: 최종적으로 하나의 보상 값(스칼라)을 출력하기 위한 차원.
        value_head = nn.Linear(model.config.n_embd, 1)
        super().__init__(model, value_head, lora_rank, lora_train_bias)

        if pretrained is not None:
            self.model = model
            self.pretrained = pretrained

    def save_pretrained(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        torch.save(self.state_dict(), os.path.join(dir, 'reward_model.pt'))
        if self.pretrained is not None:
            self.model.save_pretrained(dir)
    
    # 로딩 메서드 추가
    @classmethod
    def from_custom_pretrained(cls, save_directory, device='cpu', **kwargs):
        model = cls(**kwargs)
        state_dict = torch.load(os.path.join(save_directory, 'reward_model.pt'), map_location=device)
        model.load_state_dict(state_dict)
        return model.to(device)

@print_func_name
def run_reward_model(cfg, tokenizer):
    clear_device_cache()
    rm_model_path = os.path.join(cfg['rm_saved_dir'], 'reward_model.pt')
    
    # 모델 학습 여부에 따른 분기 처리
    if not os.path.exists(rm_model_path):
        with NaiveStrategy().model_init_context():
            model = GPTRM_custom(pretrained=cfg['model_name'],
                                lora_rank=0, tokenizer=tokenizer).to(cfg['device'])

        # 1. 데이터셋 구성 (Pairwise Ranking)
        list_data_dict = load_jsonl(cfg['data_path_2_RM'])
        total_data_ranking2chosen = prepare_pairwise_data(list_data_dict)

        # print('------- 1. ---------------------------------------------')
        # print(f'before data num: {len(list_data_dict)}')
        # print(f'after  data num: {len(total_data_ranking2chosen)}')
        # print(f'data example: {total_data_ranking2chosen[45]}')

        # 2. 데이터 셔플링 및 분할
        random.seed(230319)
        random.shuffle(total_data_ranking2chosen)
        print(total_data_ranking2chosen[45])

        train_data = total_data_ranking2chosen[:1000]
        eval_data = total_data_ranking2chosen[1000:1200]

        print(len(train_data))
        print(len(eval_data))

        # 3. RewardDataset 생성
        # * 입력 데이터 처리, 토큰화, 데이터셋 생성
        train_dataset = RewardDataset(train_data, tokenizer, 512)
        eval_dataset = RewardDataset(eval_data, tokenizer, 512)

        # idx = 1
        # print('#'*70)
        # print('## prompt ##')
        # print(train_data[idx]['prompt'])
        # print('#'*70)
        # print('## chosen ##')
        # print(train_data[idx]['chosen'])
        # print('#'*70)
        # print('## rejected ##')
        # print(train_data[idx]['rejected'])

        # 4. Reward Model 학습
        trainer = RewardModelTrainer(model=model,
                                strategy=NaiveStrategy(),
                                optim=torch.optim.Adam(model.parameters(), lr=5e-5),
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                batch_size=cfg['rm_batch_size'],
                                max_epochs=cfg['rm_max_epochs'])
        trainer.fit(use_lora=0)
        model.save_pretrained(cfg['rm_saved_dir'])
    else:
        # 5. 기존 모델 로드 (권장 방법 사용)
        with NaiveStrategy().model_init_context():
            model = GPTRM_custom.from_custom_pretrained(
                cfg['rm_saved_dir'], 
                device=cfg['device'], 
                pretrained=cfg['model_name'], 
                lora_rank=0, 
                tokenizer=tokenizer
            )

    # 6. 추론 테스트
    def inference_RM(input_text):
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(cfg['device'])
        output = model(input_ids)
        output_reward = output.cpu().detach().numpy()[0]

        print(f'input: {input_text}')
        print(f'reward score: {output_reward:.1f}')

        return output_reward

    test_texts = [
        '인공지능은 똥멍청이 입니다',
        '인공지능(AI)은 컴퓨터에서 음성 및 작성된 언어를 보고 이해하고 번역하고 데이터를 분석하고 추천하는 기능을 포함하여 다양한 고급 기능을 수행할 수 있는 일련의 기술입니다.',
        "인공지능(AI)은 컴퓨터에서 음성 및 작성된 언어를 보고 이해하고 번역하고 데이터를 분석하고 추천하는 기능을 포함하여 다양한 고급 기능을 수행할 수 있는 일련의 기술입니다. AI는 현대적인 컴퓨팅 혁신에서 중추적인 역할을 하며 개인과 비즈니스의 가치를 창출합니다. 예를 들어 광학 문자 인식(OCR)은 AI를 사용해 이미지 및 문서에서 텍스트 및 데이터를 추출하고, 구조화되지 않은 콘텐츠를 비즈니스에 바로 사용할 수 있게 만들고, 유용한 정보를 창출합니다.",
        "인공지능은 일반적으로 인간의 지능이 필요하거나 인간이 분석할 수 있는 것보다 규모가 큰 데이터를 포함하는 방식으로 추론, 학습 및 행동할 수 있는 컴퓨터 및 기계를 구축하는 것과 관련된 과학 분야입니다. AI는 컴퓨터 공학, 데이터 분석 및 통계, 하드웨어 및 소프트웨어 엔지니어링, 언어학, 신경 과학은 물론 철학과 심리학을 포함하여 여러 학문을 포괄하는 광범위한 분야입니다. 비즈니스의 운영 수준에서 AI는 주로 머신러닝과 딥 러닝을 기반으로 하는 기술 모음으로, 데이터 분석, 예상 및 예측, 객체 분류, 자연어 처리, 추천, 지능형 데이터 가져오기 등을 수행할 수 있습니다."
    ]

    for text in test_texts:
        inference_RM(text)

    # input text가 더 좋아질수록 reward score가 점진적으로 상승하나요?
    # 각 reward score 값이 적절해 보이시나요?
    # reward score가 음수가 된다는 건 어떤 의미일까요?
    # 그 전에 reward score가 음수도 될 수 있도록 하려면 어떻게 해야 할까요?
    # RM의 출력인 reward score가 scalar가 되도록 하는 게 왜 중요할까요?
    # RLHF의 마지막 단계인 PPO 학습을 통해 살펴보도록 하겠습니다.
    # 여기서도 메모리 관리를 위해 한 번더 캐시를 비우고 넘어가겠습니다.

    clear_device_cache()

@print_func_name
def run_ppo(cfg, model, tokenizer):
    """
    Proximal Policy Optimization (PPO)를 이용해 텍스트 생성 모델을 강화학습(RL)으로 미세 조정하는 함수.
    이 함수는 SFT(Supervised Fine-Tuning)된 모델과 RM(Reward Model)을 불러와, 
    RM의 보상을 최대화하도록 언어 모델을 훈련시킵니다.
    """

    device = cfg['device']

    # 1. 4개 모델 준비 
    with NaiveStrategy().model_init_context():
        # Policy Network(Actor): 현재 학습 대상. 프롬프트가 주어지면 응답을 생성합니다.
        # 이미 PPO 학습된 모델이 있다면 재사용하고, 없다면 SFT 모델을 불러옵니다.
        if os.path.exists(cfg['ppo_saved_dir']):
            print(f"\n>>> Loading pre-trained PPO model from {cfg['ppo_saved_dir']}")
            actor = GPTActor(pretrained=cfg['ppo_saved_dir'], lora_rank=0).to(device)
        else:
            print(f">>> Loading SFT model for PPO training from {cfg['sft_saved_dir']}")
            actor = GPTActor(pretrained=cfg['sft_saved_dir'], lora_rank=0).to(device)

        # Value Network(Critic): 입력된 상태(텍스트 시퀀스)의 가치(Value)를 예측합니다.
        # Actor가 더 나은 방향으로 업데이트되도록 Baseline을 제공하여 분산을 줄이는 역할을 합니다.
        critic = GPTCritic(pretrained=cfg['rm_saved_dir'], lora_rank=0).to(device)

        # Reference Model(Initial Model): 학습 과정에서 Actor 모델이 원래 가진 언어 구사 능력을 잃거나 
        # 너무 "Reward Hacking(점수만 높게 받기 위해 문맥에 맞지 않는 말도 안되는 텍스트 생성)" 하는 것을 방지하기 위해 
        # KL-Divergence 페널티를 계산할 때 기준이 되는 모델입니다. SFT 모델(초기 Actor)을 복사해 사용하며 파라미터 업데이트는 되지 않습니다.
        initial_model = deepcopy(actor)
        
        # Reward Model (보상 모델): 응답의 최종 점수를 매기는 판사 역할을 합니다 (Critic과 가중치 공유/복사)
        reward_model = RewardModel(deepcopy(critic.model), deepcopy(critic.value_head)).to(device)

    # Actor와 Critic은 각각 Optimizer를 가집니다 (PPO 정책 학습용)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=5e-6)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=5e-6)

    (actor, actor_optim), (critic, critic_optim), reward_model, initial_model = NaiveStrategy().prepare(
                                (actor, actor_optim), (critic, critic_optim), reward_model, initial_model)


    # 2. PPO 학습에 쓸 데이터셋(프롬프트만 있는 데이터) 로드 및 토크나이징 함수 정의
    list_data_dict = load_jsonl(cfg['data_path_3_PPO'])
    list_prompt = [tmp['prompt'] for tmp in list_data_dict]

    def tokenize_fn(texts):
        # 입력 텍스트(프롬프트)를 배 단위로 처리할 때 사이즈를 맞추기 위해 Padding 및 Truncation
        batch = tokenizer(texts, return_tensors='pt', max_length=96, padding=True, truncation=True)

        if str(device) == 'cuda' or str(device) == 'xpu':
            return {k: v.to(device) for k, v in batch.items()}

        return {k: v for k, v in batch.items()}

    # 3. PPO 학습 진행
    # 원본 : chatgpt/trainer/ppo.py 구조 활용
    # PPO의 loss function : policy loss와 value loss로 나뉩니다 (chatgpt/models/loss.py)

    if not os.path.exists(os.path.join(cfg['ppo_saved_dir'], 'config.json')):
        if not os.path.exists(cfg['ppo_saved_dir']):
            os.makedirs(cfg['ppo_saved_dir'])

        print(">>> Starting PPO training...")
        # PPOTrainer 초기화 및 하이퍼파라미터 설정
        trainer = PPOTrainer(NaiveStrategy(),
                     actor,
                     critic,
                     reward_model,
                     initial_model,
                     actor_optim,
                     critic_optim,
                     max_epochs=1,
                     train_batch_size=8,
                     tokenizer=tokenize_fn,
                     max_length=128,          # 생성될 전체 시퀀스(프롬프트+답변) 최대 길이
                     do_sample=True,          # 생성 다양성을 위해 샘플링 적용
                     temperature=1.0,         # Softmax 온도 계수
                     top_k=50,                # 샘플링 가능한 후보 단어 수 제한
                     pad_token_id=tokenizer.pad_token_id,
                     eos_token_id=tokenizer.eos_token_id
        )

        # num_episodes: 전체 데이터(프롬프트 집합)를 몇 번 돌지 지정
        trainer.fit(list_prompt,
                num_episodes=10,
                max_timesteps=3,              # 각 프롬프트에 대해 텍스트를 생성하여 환경과 상호작용하는 스텝 수
                update_timesteps=3)           # 쌓인 경험(Experience)을 바탕으로 네트워크를 업데이트하는 주기
    
        # 학습된 최종 Policy Network 저장
        actor.model.save_pretrained(cfg['ppo_saved_dir'])
        print(">>> PPO training completed and model saved.")
    else:
        print(">>> PPO training skipped as pre-trained model was loaded.")

    # 4. PPO 모델 추론(Inference) 단계: 기본 테스트 프롬프트들에 대해 답변을 잘 생성하는지 실험
    list_prompt = [PROMPT_TEMPLATE.format_map({'prompt': tmp}) for tmp in DEFAULT_TEST_PROMPTS]

    outputs = generate_custom(list_prompt, actor, tokenizer, cfg['device'])
    # for output in outputs:
    #     print(output) 모델입니다. 프롬프트에 대해 답변을 생성합니다.
#   Critic (가치 모델): 특정 상태의 가치($V$)를 예측하여 Actor의 업데이트를 돕습니다.
#   Initial Model (참조 모델): 학습 중 Actor가 기존 언어 능력을 잃고 너무 이상하게 변하지 않도록(KL Divergence 제어) 기준점이 되어주는 모델입니다.
#   Reward Model (보상 모델): 생성된 답변이 얼마나 좋은지 점수를 매기는 판사 역할을 합니다.
# -----------------------------------------------------------------------------
@print_func_name
def run_ppo(cfg, model, tokenizer):

    device = cfg['device']

    # 1. 4개 모델 준비 
    with NaiveStrategy().model_init_context():
        if os.path.exists(cfg['ppo_saved_dir']):
            print(f"\n>>> Loading pre-trained PPO model from {cfg['ppo_saved_dir']}")
            actor = GPTActor(pretrained=cfg['ppo_saved_dir'], lora_rank=0).to(device)
        else:
            print(f">>> Loading SFT model for PPO training from {cfg['sft_saved_dir']}")
            actor = GPTActor(pretrained=cfg['sft_saved_dir'], lora_rank=0).to(device)

        critic = GPTCritic(pretrained=cfg['rm_saved_dir'], lora_rank=0).to(device)

        initial_model = deepcopy(actor)
        reward_model = RewardModel(deepcopy(critic.model), deepcopy(critic.value_head)).to(device)


    actor_optim = torch.optim.Adam(actor.parameters(), lr=5e-6)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=5e-6)

    (actor, actor_optim), (critic, critic_optim), reward_model, initial_model = NaiveStrategy().prepare(
                                (actor, actor_optim), (critic, critic_optim), reward_model, initial_model)

    # PPO 학습에 쓸 데이터를 토크나이징
    list_data_dict = load_jsonl(cfg['data_path_3_PPO'])
    list_prompt = [tmp['prompt'] for tmp in list_data_dict]

    def tokenize_fn(texts):
        batch = tokenizer(texts, return_tensors='pt', max_length=96, padding=True, truncation=True)

        if str(device) == 'cuda' or str(device) == 'xpu':
            return {k: v.to(device) for k, v in batch.items()}

        return {k: v for k, v in batch.items()}

    # PPO 학습
    if not os.path.exists(os.path.join(cfg['ppo_saved_dir'], 'config.json')):
        if not os.path.exists(cfg['ppo_saved_dir']):
            os.makedirs(cfg['ppo_saved_dir'])

        print(">>> Starting PPO training...")
        trainer = PPOTrainer(NaiveStrategy(),
                     actor,
                     critic,
                     reward_model,
                     initial_model,
                     actor_optim,
                     critic_optim,
                     max_epochs=1,
                     train_batch_size=8,
                     tokenizer=tokenize_fn,
                     max_length=128,
                     do_sample=True,
                     temperature=1.0,
                     top_k=50,
                     pad_token_id=tokenizer.pad_token_id,
                     eos_token_id=tokenizer.eos_token_id
        )

        trainer.fit(list_prompt,
                num_episodes=10,
                max_timesteps=3,
                update_timesteps=3)
    
        actor.model.save_pretrained(cfg['ppo_saved_dir'])
        print(">>> PPO training completed and model saved.")
    else:
        print(">>> PPO training skipped as pre-trained model was loaded.")

    list_prompt = [PROMPT_TEMPLATE.format_map({'prompt': tmp}) for tmp in DEFAULT_TEST_PROMPTS]

    outputs = generate_custom(list_prompt, actor, tokenizer, cfg['device'])
    for output in outputs:
        print(output)

def run_baseline():
    """ NODE baseline: 그냥 찍는 수준 """

    model_name = "skt/kogpt2-base-v2"                       # V
    tcname = model_name.replace("/", "-") + "_baseline"

    try:
        root_path = os.path.dirname(os.path.abspath(__file__))
    except:
        root_path = os.getcwd()
        
    cfg = {
        "model_name": model_name,                       # V
        "sft_tokenizer": model_name,
        "device": torch.device("xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"),
        "root_path": root_path,

        "sft_output_dir": os.path.join(root_path, "test", tcname),
        "sft_saved_dir": os.path.join(root_path, "models", tcname + "_output_1_SFT"),
        "rm_saved_dir": os.path.join(root_path, "models", tcname + "_output_2_RM"),
        "ppo_saved_dir": os.path.join(root_path, "models", tcname + "_output_3_PPO"),

        "data_path_1_SFT": os.path.join(root_path, "KoChatGPT/data_kochatgpt/kochatgpt_1_SFT.jsonl"),
        "data_path_2_RM": os.path.join(root_path, "KoChatGPT/data_kochatgpt/kochatgpt_2_RM.jsonl"),
        "data_path_3_PPO": os.path.join(root_path, "KoChatGPT/data_kochatgpt/kochatgpt_3_PPO.jsonl"),

        "sft_num_train_epochs": 1,              # V
        "sft_per_device_train_batch_size": 4,   # V
        "sft_per_device_eval_batch_size": 4,    # v 
        "sft_warmup_steps": 5,                  # V 해당 step 동안 학습률을 0 -> 목표값까지 증가
        "sft_prediction_loss_only": True,       # V 학습/평가 시 손실값(Loss)만 계산할지 예측값(Logits/Metrics)까지 계산할지 - 일단 Fix 
        "sft_fp16": False,                      # 메모리 절약

        "rm_batch_size": 4,                     # V
        "rm_max_epochs": 1,                     # V

         "rm_seed": 230319, 
         "rm_max_train_samples": 1000, 
         "rm_max_len": 512, 
         "rm_lr": 5e-5 
    }

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"]).to(cfg["device"])
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        cfg["model_name"],
        bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>',
        padding_side="right",
        model_max_length=512,
    )

    show_info(cfg)
    run_sft(cfg, model, tokenizer)
    run_reward_model(cfg, tokenizer)
    run_ppo(cfg, model, tokenizer)
    run_evaluation(cfg)


def run_case1():
    """ SFT 관련 param 조정 """

    model_name = "skt/kogpt2-base-v2"
    tcname = model_name.replace("/", "-") + "_case1"

    try:
        root_path = os.path.dirname(os.path.abspath(__file__))
    except:
        root_path = os.getcwd()
        
    cfg = {
        "model_name": model_name,
        "sft_tokenizer": model_name,
        "device": torch.device("xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"),
        "root_path": root_path,

        "sft_output_dir": os.path.join(root_path, "test", tcname),
        "sft_saved_dir": os.path.join(root_path, "models", tcname + "_output_1_SFT"),
        "rm_saved_dir": os.path.join(root_path, "models", tcname + "_output_2_RM"),
        "ppo_saved_dir": os.path.join(root_path, "models", tcname + "_output_3_PPO"),

        "data_path_1_SFT": os.path.join(root_path, "KoChatGPT/data_kochatgpt/kochatgpt_1_SFT.jsonl"),
        "data_path_2_RM": os.path.join(root_path, "KoChatGPT/data_kochatgpt/kochatgpt_2_RM.jsonl"),
        "data_path_3_PPO": os.path.join(root_path, "KoChatGPT/data_kochatgpt/kochatgpt_3_PPO.jsonl"),

        "sft_num_train_epochs": 3,              # 1 -> 3
        "sft_per_device_train_batch_size": 8,   # 4 -> 8 : batch size 늘리면 효과는 좋음. ->  gradient가 안정적으로 계산되기 때문
        "sft_per_device_eval_batch_size": 8,    # 4 -> 8 : eval batch size는 메모리 사용량에 영향이 없음. 왜냐하면 eval은 학습이 아니기 때문 ?
        "sft_warmup_steps": 10,                 # 5 -> 10 : 
        "sft_prediction_loss_only": True,
        "sft_fp16": False,

        "rm_batch_size": 4,
        "rm_max_epochs": 1,

        "rm_seed": 230319, 
        "rm_max_train_samples": 1000, 
        "rm_max_len": 512, 
        "rm_lr": 5e-5 
    }

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"]).to(cfg["device"])
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        cfg["model_name"],
        bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>',
        padding_side="right",
        model_max_length=512,
    )

    show_info(cfg)
    run_sft(cfg, model, tokenizer)
    run_reward_model(cfg, tokenizer)
    run_ppo(cfg, model, tokenizer)
    run_evaluation(cfg)


if __name__ == "__main__":
    run_baseline()
    run_case1()

