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
# $ pip install datasets loralib trl
# # pip install colossalai --upgrade

# 공지
# 1. 토크나이저가 오류
#   허깅페이스에서 한국어 토크나이저를 여럿 제공할텐데요, 적당한 걸 찾아서 쓰시면 되겠습니다
#   일단 저는 "monologg/koelectra-base-v3-discriminator" 이거로 교체를 했구요!
#   이것보다 더 좋은 토크나이저들이 있을거에요

#   응답이 AI...와 같이 깨져서 나오는 현상은
#       skt/kogpt2-base-v2 모델과 토크나이저 설정이 맞지 않거나,
#       특수 토큰 설정 과정에서 한국어 데딩 인코딩이 꼬였을 때 발생합니다.
#   이를 해결하기 위해 KoGPT-2의 공식적인 권장 방식인 PreTrainedTokenizerFast를 사용하고,
#   모델에 최적화된 특수 토큰(bos, eos, unk, pad, mask)을 정확히 설정하도록 변경



# -----------------------------------------------------------------------------
# 내가 궁금한 것, 여기서 중요한 것
# 1. Ref 코드가 한 것과 Node 코드가 한 것.
# 2. pre-trained - fine-tuning - RLHF - reward model 의 학습 과정
# 3. 언어 모델의 정량적, 정성적 평가 방법
# 4.
# -----------------------------------------------------------------------------

import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset


import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy
import json

from typing import Optional, Dict, Sequence
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
import pandas as pd
import numpy
import json
import logging
import copy

try:
    root_path = os.path.dirname(os.path.abspath(__file__))
except:
    root_path = os.getcwd()


def step_01_modify_files():
    modifications = [
        {
            # "file": "/content/chatgpt/trainer/callbacks/save_checkpoint.py",
            "file": f"{root_path}/chatgpt/trainer/callbacks/save_checkpoint.py",
            "changes": [
                {
                    "line": 3,
                    "old": "from chatgpt.trainer.strategies import ColossalAIStrategy, Strategy",
                    "new": "from chatgpt.trainer.strategies import Strategy"
                },
                {
                    "line": 71,
                    "old": "only_rank0 = not isinstance(self.strategy, ColossalAIStrategy)",
                    "new": "            only_rank0 = not isinstance(self.strategy)"
                },
            ],
        },
        {
            "file": f"{root_path}/chatgpt/trainer/strategies/__init__.py",
            "changes": [
                {
                    "line": 1,
                    "old": "from .colossalai import ColossalAIStrategy",
                    "new": ""
                },
                {
                    "line": 5,
                    "old": "__all__ = ['Strategy', 'NaiveStrategy', 'DDPStrategy', 'ColossalAIStrategy']",
                    "new": "__all__ = ['Strategy', 'NaiveStrategy', 'DDPStrategy']"
                },
            ],
        },
        {
            "file": f"{root_path}/chatgpt/dataset/reward_dataset.py",
            "changes": [
                {
                    "line": 3,
                    "old": "from tqdm import tqdm",
                    "new": "from tqdm.notebook import tqdm"},
            ],
        },
        {
            "file": f"{root_path}/chatgpt/trainer/base.py",
            "changes": [
                {
                    "line": 8,
                    "old": "from tqdm import tqdm",
                    "new": "from tqdm.notebook import tqdm"
                },
            ]
        },
        {
            "file": f"{root_path}/chatgpt/trainer/rm.py",
            "changes": [
                {
                    "line": 8,
                    "old": "from tqdm import tqdm",
                    "new": "from tqdm.notebook import tqdm"
                    },
            ]
        }
    ]

    def modify_file(file_path, changes):
        if not os.path.exists(file_path):
            print(f"ERROR: file not found : {file_path}")
            return

        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        modified = False

        for change in changes:
            line_index = change["line"]
            if 0 <= line_index < len(lines):
                if lines[line_index].strip() == change["old"]:
                    lines[line_index] = change["new"] + "\n"
                    modified = True
                else:
                    print(f"ERROR: {file_path} file {change['line']}th line not match")
                    print(f"   expected: {change['old']}")
                    print(f"   actual: {lines[line_index].strip()}")

        if modified:
            with open(file_path, "w", encoding="utf-8") as file:
                file.writelines(lines)
            print(f"SUCCESS: {file_path} modified")
        else:
            print(f"ERROR: {file_path} no modification")

    for mod in modifications:
        modify_file(mod["file"], mod["changes"])

step_01_modify_files()


from chatgpt.dataset import RewardDataset
from chatgpt.models.base import RewardModel
from chatgpt.trainer.strategies import NaiveStrategy
from chatgpt.trainer.rm import RewardModelTrainer
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

import random



def step_02_show_info(cfg):
    print("Torch version:{}".format(torch.__version__)) # Torch version:1.12.1
    # print("Cuda version: {}".format(torch.version.cuda)) # Cuda version: 11.3
    print("Device: {}".format(cfg['device']))
    print("transformers version: {}".format(transformers.__version__)) # transformers 4.28.0
    
    if "cuda" in str(cfg['device']):
        print("GPU available: {}".format(torch.cuda.is_available()))
    elif "xpu" in str(cfg['device']):
        print("XPU available: {}".format(torch.xpu.is_available()))
    else:
        print("Using CPU")

# -----------------------------------------------------------------------------
# 2. Base model and Dataset for RLHF
def show_base_model_and_dataset(cfg, model, tokenizer):

    # 베이스라인 모델 - kogpt-2의 일반적인 성능을 확인하기
    print('--- 1 ------------------------------------------------')
    r = tokenizer.model_max_length
    print(r)   # 입력 최대 토큰 수

    r = model.config.n_positions
    print(r)    # 모델이 입력받아 처리할 수 있는 최대 토큰 수 ?

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
def show_sft_and_rm_dataset(cfg):

    # SFT DataSet - prompt, completion, tokens
    data_path_1_SFT = f'{cfg["root_path"]}/KoChatGPT/data_kochatgpt/kochatgpt_1_SFT.jsonl'
    with open(data_path_1_SFT, "r", encoding='utf-8-sig') as json_file:
        list_data_dict = json.load(json_file)

    r = f'{len(list_data_dict):,}'
    print(r)
    r = list_data_dict[:3]
    print(r)

    # RM DataSet - prompt, completion_0, 1, 2, ranking
    data_path_2_RM = f'{cfg["root_path"]}/KoChatGPT/data_kochatgpt/kochatgpt_2_RM.jsonl'
    with open(data_path_2_RM, "r", encoding='utf-8-sig') as json_file:
        list_data_dict = json.load(json_file)

    r = f'{len(list_data_dict):,}'
    print(r)
    r = list_data_dict[:3]
    print(r)

    # PPO (Proximal Policy Optimization) 학습에 쓰일 데이터 준비 - prompt only
    data_path_3_PPO = f'{cfg["root_path"]}/KoChatGPT/data_kochatgpt/kochatgpt_3_PPO.jsonl'
    with open(data_path_3_PPO, "r", encoding='utf-8-sig') as json_file:
        list_data_dict = json.load(json_file)

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

        with open(data_path_1_SFT, "r", encoding='utf-8-sig') as json_file:
            list_data_dict = json.load(json_file)

        PROMPT_DICT = {
            "prompt_input": (
                "### Instruction(명령어):\n{prompt}\n\n### Response(응답):"
            )
        }

        prompt_input = PROMPT_DICT["prompt_input"]

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

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # padding_value = -100 ?? 의미는 ?
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value= -100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

import functools

def print_func_name(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"-------------------- {func.__name__}() ----------------------------------------")
        return func(*args, **kwargs)
    return wrapper

# SFT(Supervised Fine-Tuning) 수행 후 결과 확인
@print_func_name
def run_sft(cfg, model, tokenizer):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.empty_cache()
        
    # 학습용 배치 Batch 생성기
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    train_dataset = SFT_dataset(
        data_path_1_SFT=f'{cfg["root_path"]}/KoChatGPT/data_kochatgpt/kochatgpt_1_SFT.jsonl',
        tokenizer=tokenizer
    )

    # r = f'in : {train_dataset.input_ids[0]}'
    # print(r)
    # r = f'out: {train_dataset.labels[0]}'
    # print(r)

    # # train_dataset.input_ids[0]를 디코딩 하는 함수
    # r = tokenizer.decode(train_dataset.input_ids[0])
    # print(r)

    # trainer 클래스 정의
    # https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/trainer#transformers.TrainingArguments
    #
    # 변경 추가등. .. 여기서 만지작 해야 한다. !!!
    training_args = transformers.TrainingArguments(
        output_dir=cfg['sft_output_dir'],
        num_train_epochs=cfg['sft_num_train_epochs'],           # epoch 수
        per_device_train_batch_size=cfg['sft_per_device_train_batch_size'],
        per_device_eval_batch_size=cfg['sft_per_device_eval_batch_size'],
        warmup_steps=cfg['sft_warmup_steps'],
        prediction_loss_only=cfg['sft_prediction_loss_only'],   
        fp16 = cfg['sft_fp16']                                  # 메모리 절약
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    # if not os.path.exists(f"{cfg['sft_saved_dir']}"):
    if not os.path.exists(os.path.join(cfg['sft_saved_dir'], 'config.json')):
        if not os.path.exists(cfg['sft_saved_dir']):
            os.makedirs(cfg['sft_saved_dir'])
        trainer.train() # 학습 시작
        model.save_pretrained(cfg['sft_saved_dir'])
    else:
        print(f"SFT model already exists: {cfg['sft_saved_dir']}")

    # 문장 생성 능력 확인위한 pipleline generator 생성
    # Text generation strategies
    generator = transformers.pipeline(
        'text-generation',
        model=cfg['sft_saved_dir'],
        tokenizer=tokenizer
    )

    # 추론 및 생성 전략
    generation_args = dict(
        num_beams=4, # 빔 서치의 너비. 클수록 더 좋은 문장을 생성할 확률이 높지만 속도가 느려짐.
        repetition_penalty=2.0, # 반복 패널티. 1.0보다 크면 반복을 억제함.
        no_repeat_ngram_size=4, # 값을 2~3으로 조정하면 단어 뭉치가 반복되는 것을 더 세밀하게 막을 수 있다.
        eos_token_id=375,       # 문장 생성 종료 토큰
        max_new_tokens=64,      # 생성할 최대 토큰 수
        do_sample=True,         # sampling을 통해 더 다양한 문장을 생성한다.
        top_k=50,               # 값을 40~50 정도로 설정하면 모델이 다음 단어를 선택할 때 확률이 높은 상위 50개 단어 중에서만 고르게 된다. 
                                # (낮을수록 일관성↑, 높을수록 다양성↑)
        early_stopping=True     # 조기 종료
    )

    PROMPT_DICT = {
        "prompt_input": (
            "*** Instruction(명령어):\n{prompt}\n\n*** Response(응답):"
        )
    }

    list_prompt = ['불고기용 고기 한우에요?',
                '리처드 닉슨이 43대 부통령직을 수행한 년도는?',
                '시카고 오헤어 국제공항은 어디에 있어?',
                '오늘 미세먼지 어때?']

    list_prompt = [PROMPT_DICT['prompt_input'].format_map({'prompt' : tmp}) for tmp in list_prompt]

    list_result = generator(list_prompt, **generation_args)
    for prompt, result in zip(list_prompt, list_result):
        print()
        print((result[0]['generated_text']))


# -----------------------------------------------------------------------------
# 4. Reward Model
# -----------------------------------------------------------------------------
class GPTRM_custom(RewardModel):
    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[GPT2Config] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none',
                 tokenizer=None) -> None:
        if pretrained is not None:
            model = GPT2Model.from_pretrained(pretrained)
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

def test_04():
    # 모델과 토크나이저 로드
    # with구문의 NaiveStrategy는 chatgpt/trainer/strategies/base 모듈에서 정의된
    # Strategy클래스를 상속한 NaiveStrategy클래스이다.

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.empty_cache()

    model = AutoModelForCausalLM.from_pretrained('skt/kogpt2-base-v2')
    tokenizer = AutoTokenizer.from_pretrained(
        'skt/kogpt2-base-v2',
        bos_token='</s>', eos_token='</s>', unk_token='</s>', pad_token='</s>',
        padding_side="right",
        model_max_length=512,
    )

    with NaiveStrategy().model_init_context():
        model = GPTRM_custom(pretrained='skt/kogpt2-base-v2',
                            lora_rank=0, tokenizer=tokenizer).to(cfg['device'])
                            # lora_rank=0, tokenizer=tokenizer).cuda()

    # RM을 훈련시킬 때 사용할 ranking dataset 생성
    with open('KoChatGPT/data_kochatgpt/kochatgpt_2_RM.jsonl', "r", encoding='utf-8-sig') as json_file:
        list_data_dict = json.load(json_file)

    total_data_ranking2chosen = []
    for tmp in list_data_dict:
        one_data_ranking2chosen = []

        data = {}
        data['prompt'] = tmp['prompt']
        if tmp['ranking'][0] < tmp['ranking'][1]:
            data['chosen'] = tmp['completion_0']
            data['rejected'] = tmp['completion_1']
        else:
            data['chosen'] = tmp['completion_1']
            data['rejected'] = tmp['completion_0']
        one_data_ranking2chosen.append(data)

        data = {}
        data['prompt'] = tmp['prompt']
        if tmp['ranking'][0] < tmp['ranking'][2]:
            data['chosen'] = tmp['completion_0']
            data['rejected'] = tmp['completion_2']
        else:
            data['chosen'] = tmp['completion_2']
        data['rejected'] = tmp['completion_0']
        one_data_ranking2chosen.append(data)

        data = {}
        data['prompt'] = tmp['prompt']
        if tmp['ranking'][1] < tmp['ranking'][2]:
            data['chosen'] = tmp['completion_1']
            data['rejected'] = tmp['completion_2']
        else:
            data['chosen'] = tmp['completion_2']
            data['rejected'] = tmp['completion_1']
        one_data_ranking2chosen.append(data)
        total_data_ranking2chosen.extend(one_data_ranking2chosen)

    print('------- 1. ---------------------------------------------')
    print(f'before data num: {len(list_data_dict)}')
    print(f'after  data num: {len(total_data_ranking2chosen)}')
    print(f'data example: {total_data_ranking2chosen[45]}')

    """
    # kochatgpt_2_RM.jsonl 은
    #   chatGPT, davinch, ada 세개 모델에 같은 prompt를 주고 얻은 세 답변을
    #   순서대로 good, bad, worst로 간주해
    #   순서를 뒤섞어 completion_0, completion_1, completion_2 세 키에 할당하여 만든 데이터셋입니다.
    #   이러면 chosen과 resjected에 각각
    #   completion_0, completion_1, completion_2 세개 답변이 가능한 모든 조합으로 들어가게 되어
    #       chosen에 worst 답변이 들어가고
    #       rejected에 good답변이 들어간 데이터도 만들어집니다.

    # 위와 같이 ranking dataset을 만들면 RM의 loss는 어떻게 계산이 되는 걸까요?
    # RM의 loss function은 pairwiseloss라는 이름으로 설계되어 있습니다.
    # 아래 pairwiseloss 코드를 첨부했습니다.
    # 원본 코드는 chatgpt/models 폴더의 loss.py 를 확인해보세요.

    # Q. 위 코드블럭에서 probs = torch.sigmoid(chosen_reward - reject_reward) 코드를 찾아보세요.
    #   chosen_reward - reject_reward 식은 어떤 연산을 의미하나요? loss = -log_probs.mean() 코드는 무엇을 최대화하는 연산으로 해석할 수 있을까요?
    # A. `chosen_reward` - reject_reward는 선택된(chosen) 보상과 거부된(reject) 보상 간의 차이를 계산하는 연산입니다.
    # 이 차이가 클수록 선택된 샘플이 거부된 샘플보다 높은 보상을 받았다는 것을 의미하죠.
    # 여기서 `torch.sigmoid(chosen_reward - reject_reward)`를 적용하면, 두 보상의 차이를 확률 값(0과 1 사이)으로 변환하게 됩니다.
    # 이 확률 값은 선택된 샘플이 거부된 샘플보다 더 좋은 결과를 낼 확률로 해석할 수 있습니다.
    # `loss = -log_probs.mean()` 코드는 선택된 샘플이 거부된 샘플보다 더 좋은 결과를 내는 log 확률을 최대화하는 방향으로 작동합니다.
    # class PairWiseLoss(nn.Module):
    #     def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
    #         probs = torch.sigmoid(chosen_reward - reject_reward)
    #         log_probs = torch.log(probs)
    #         loss = -log_probs.mean()
    #         return loss

    # total_data_ranking2chosen = []

    # for tmp in list_data_dict:
    #      prompt = tmp['prompt']
    #      ranking = tmp['ranking']

    #      for index in range(1, len(ranking)):
    #          n = ranking[0]
    #          m = ranking[index]


    #          data = {
    #              'prompt': prompt,
    #              'chosen': tmp['completion_{}'.format(n)],
    #              'rejected': tmp['completion_{}'.format(m)]
    #          }

    #          total_data_ranking2chosen.append(data)
    # Q. 위 코드대로 ranking dataset 함수를 수정하게 되면 ranking data가 어떻게 만들어지게 되나요? 둘의 차이를 비교하고 어떤 데이터셋을 사용하는게 더 적절한지 이야기해봅시다.
    # A. 힌트
    # 힌트
    # **기존 코드** (단순 ranking2chosen)
    # ranking의 첫 번째 후보(최고 순위)를 항상 chosen으로 사용하고, 나머지 후보들을 rejected로 하는 방식입니다.
    # - 예를 들어, ranking이 [A, B, C]라면 (A가 최고라고 가정)
    #    - (prompt, chosen=A, rejected=B)
    #    - (prompt, chosen=A, rejected=C)
    # 즉, 각 프롬프트당 2개의 pair가 생성됩니다.

    # **비교 코드** (전체 쌍 생성):
    # 후보들 간의 모든 쌍을 비교하여, 각 pair마다 어느 쪽이 더 좋은지 ranking 값을 비교합니다.
    # - ranking이 [A, B, C]라면
    #    - (prompt, chosen=A, rejected=B)
    #    - (prompt, chosen=A, rejected=C)
    #    - (prompt, chosen=B, rejected=C)
    # 즉, 각 프롬프트당 3개의 pair가 생성됩니다.

    # 만약 데이터의 ranking이 신뢰할 수 있고, 후보들 간의 미세한 차이를 모델이 잘 반영해야 한다면, 전체 쌍 생성 방식이 더 풍부한 학습 신호를 제공할 수 있습니다.
    # 반면에, 노이즈나 불확실성이 큰 데이터라면, 단순히 최고 후보와의 비교만 사용하는 것이 더 안정적일 수 있습니다
    """

    # 완성한 ranking dataset을 shuffle한 후 훈련셋을 만들어보겠습니다.
    # 빠르게 돌려보기 위해 전체 데이터중 일부만 학습
    random.seed(230319)
    random.shuffle(total_data_ranking2chosen)
    print(total_data_ranking2chosen[45])

    train_data = total_data_ranking2chosen[:1000]
    eval_data = total_data_ranking2chosen[1000:1200]

    print(len(train_data))
    print(len(eval_data))

    # RewardDataset
    # * 입력 데이터 처리
    #       각 데이터 항목은 prompt, chosen, rejected 등의 키를 가지며,
    #       이 클래스는 해당 텍스트들을 받아서 모델 학습에 적합한 형식으로 변환합니다.
    # * 토큰화(tokenization)
    #       토큰 ID로 변환, 최대 길이, 패딩(padding) 적용
    # * 데이터셋 생성
    #       전처리된 데이터를 PyTorch의 Dataset 형식(예: torch.utils.data.Dataset)으로 생성
    train_dataset = RewardDataset(train_data, tokenizer, 512)
    eval_dataset = RewardDataset(eval_data, tokenizer, 512)

    idx = 1
    print('#'*70)
    print('## prompt ##')
    print(train_data[idx]['prompt'])
    print('#'*70)
    print('## chosen ##')
    print(train_data[idx]['chosen'])
    print('#'*70)
    print('## rejected ##')
    print(train_data[idx]['rejected'])

    # Reward Model 학습
    if not os.path.exists(os.path.join('models/output_2_RM', 'reward_model.pt')):
        trainer = RewardModelTrainer(model=model,
                                 strategy=NaiveStrategy(),
                                 optim=torch.optim.Adam(model.parameters(), lr=5e-5),
                                 train_dataset=train_dataset,
                                 eval_dataset=eval_dataset,
                                 batch_size=4,
                                 max_epochs=1)
        trainer.fit(use_lora=0)
        model.save_pretrained('models/output_2_RM')
    else:
        # 권장 방법: 커스텀 클래스로 로드하고 state_dict 적용
        with NaiveStrategy().model_init_context():
            model = GPTRM_custom(pretrained='skt/kogpt2-base-v2', lora_rank=0, tokenizer=tokenizer)
            state_dict = torch.load('models/output_2_RM/reward_model.pt', map_location=cfg['device'])
            model.load_state_dict(state_dict)
            model = model.to(cfg['device'])

    def inference_RM(input_text):
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(cfg['device'])
        output = model(input_ids)
        output_reward = output.cpu().detach().numpy()[0]

        print('input: %s\nreward score: %.1f'%(input_text, output_reward))

        return output_reward

    input_text = '인공지능은 똥멍청이 입니다'
    output_reward = inference_RM(input_text=input_text)

    input_text = '인공지능(AI)은 컴퓨터에서 음성 및 작성된 언어를 보고 이해하고 번역하고 데이터를 분석하고 추천하는 기능을 포함하여 다양한 고급 기능을 수행할 수 있는 일련의 기술입니다.'
    output_reward = inference_RM(input_text=input_text)

    input_text = "인공지능(AI)은 컴퓨터에서 음성 및 작성된 언어를 보고 이해하고 번역하고 데이터를 분석하고 추천하는 기능을 포함하여 다양한 고급 기능을 수행할 수 있는 일련의 기술입니다. AI는 현대적인 컴퓨팅 혁신에서 중추적인 역할을 하며 개인과 비즈니스의 가치를 창출합니다. 예를 들어 광학 문자 인식(OCR)은 AI를 사용해 이미지 및 문서에서 텍스트 및 데이터를 추출하고, 구조화되지 않은 콘텐츠를 비즈니스에 바로 사용할 수 있게 만들고, 유용한 정보를 창출합니다."
    output_reward = inference_RM(input_text=input_text)

    input_text = "인공지능은 일반적으로 인간의 지능이 필요하거나 인간이 분석할 수 있는 것보다 규모가 큰 데이터를 포함하는 방식으로 추론, 학습 및 행동할 수 있는 컴퓨터 및 기계를 구축하는 것과 관련된 과학 분야입니다. AI는 컴퓨터 공학, 데이터 분석 및 통계, 하드웨어 및 소프트웨어 엔지니어링, 언어학, 신경 과학은 물론 철학과 심리학을 포함하여 여러 학문을 포괄하는 광범위한 분야입니다. 비즈니스의 운영 수준에서 AI는 주로 머신러닝과 딥 러닝을 기반으로 하는 기술 모음으로, 데이터 분석, 예상 및 예측, 객체 분류, 자연어 처리, 추천, 지능형 데이터 가져오기 등을 수행할 수 있습니다."
    output_reward = inference_RM(input_text=input_text)


    # input text가 더 좋아질수록 reward score가 점진적으로 상승하나요?
    # 각 reward score 값이 적절해 보이시나요?
    # reward score가 음수가 된다는 건 어떤 의미일까요?
    # 그 전에 reward score가 음수도 될 수 있도록 하려면 어떻게 해야 할까요?
    # RM의 출력인 reward score가 scalar가 되도록 하는 게 왜 중요할까요?
    # RLHF의 마지막 단계인 PPO 학습을 통해 살펴보도록 하겠습니다.
    # 여기서도 메모리 관리를 위해 한 번더 캐시를 비우고 넘어가겠습니다.

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.empty_cache()


# -----------------------------------------------------------------------------
# 5. Proximal Policy Optimization (PPO)
# -----------------------------------------------------------------------------
from chatgpt.models.gpt import GPTActor, GPTCritic
from chatgpt.trainer import PPOTrainer
from copy import deepcopy

def test_05():
    root_path = os.path.dirname(os.path.abspath(__file__))
    device = cfg['device']
    tokenizer_name = "skt/kogpt2-base-v2"

    ppo_model_path = os.path.join(root_path, 'models/output_3_PPO')
    ppo_model_exists = os.path.exists(ppo_model_path)

    with NaiveStrategy().model_init_context():
        if ppo_model_exists:
            print(f"✅ Loading pre-trained PPO model from {ppo_model_path}")
            actor = GPTActor(pretrained=ppo_model_path, lora_rank=0).to(device)
        else:
            print(f"🚀 Loading SFT model for PPO training from {root_path}/models/output_1_SFT")
            actor = GPTActor(pretrained=f'{root_path}/models/output_1_SFT', lora_rank=0).to(device)

        critic = GPTCritic(pretrained=f'{root_path}/models/output_2_RM', lora_rank=0).to(device)


        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            tokenizer_name,
            bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>',
            padding_side="right",
            model_max_length=512,
        )

        initial_model = deepcopy(actor)
        reward_model = RewardModel(deepcopy(critic.model), deepcopy(critic.value_head)).to(device)

    # 옵티마이저, 모델 준비
    actor_optim = torch.optim.Adam(actor.parameters(), lr=5e-6)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=5e-6)

    (actor, actor_optim), (critic, critic_optim), reward_model, initial_model = NaiveStrategy().prepare(
                                (actor, actor_optim), (critic, critic_optim), reward_model, initial_model)

    # PPO 학습에 쓸 데이터를 불러와 토크나이징
    with open(f'{root_path}/KoChatGPT/data_kochatgpt/kochatgpt_3_PPO.jsonl', "r", encoding='utf-8-sig') as json_file:
        list_data_dict = json.load(json_file)
        list_prompt = [tmp['prompt'] for tmp in list_data_dict]

    def tokenize_fn(texts):
        batch = tokenizer(texts, return_tensors='pt', max_length=96, padding=True, truncation=True)

        if str(device) == 'cuda' or str(device) == 'xpu':
            return {k: v.to(device) for k, v in batch.items()}

        return {k: v for k, v in batch.items()}

    print(tokenize_fn('It takes something more than intelligence to act intelligently.'))
    len(list_prompt)


    # PPO는 별도의 PPOTrainer 클래스를 설계하여 학습해야 한다.
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

    # 원본 : chatgpt/trainer/ppo.py
    # 복잡도 : PPO > SFT, RM
    # PPO의 loss function : chatgpt/models/loss.py PolicyLoss, ValueLoss class 에서 정의

    # PPO 학습
    if not os.path.exists(os.path.join(ppo_model_path, 'config.json')):
        print("⏳ Starting PPO training...")
        trainer.fit(list_prompt,
                num_episodes=10,
                max_timesteps=3,
                update_timesteps=3)

        actor.model.save_pretrained('models/output_3_PPO')
        print("✅ PPO training completed and model saved.")
    else:
        print("⏩ PPO training skipped as pre-trained model was loaded.")


    # 드디어 SFT, RM 그리고 PPO 학습이 모두 완료되었습니다.
    # RLHF가 적용된 koGPT-2의 생성능력을 확인해볼까요?
    def generation(input_text, model):
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(cfg['device'])
        outputs = model.generate(input_ids,
                                max_length=250,
                                do_sample=True,
                                top_k=50,
                                top_p=0.95,
                                num_return_sequences=1)
        output = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]
        print()
        print(output)
        return output

    PROMPT_DICT = {
        "prompt_input": (
            "*** Instruction(명령어):\n{prompt}\n\n*** Response(응답):"
        )
    }

    list_prompt = [
        '불고기용 고기 한우에요?',
        '리처드 닉슨이 43대 부통령직을 수행한 년도는?',
        '시카고 오헤어 국제공항은 어디에 있어',
        '오늘 미세먼지 어때?']

    list_prompt = [PROMPT_DICT['prompt_input'].format_map({'prompt': tmp}) for tmp in list_prompt]

    for input_text in list_prompt:
        output = generation(input_text, actor)




cfg = {
    "model_name": "skt/kogpt2-base-v2",
    # "device": "cuda" if torch.cuda.is_available() else "cpu",
    "device": torch.device("xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"),
    "root_path": root_path,

    "sft_output_dir": root_path + "/test",
    "sft_saved_dir": root_path + "/models/output_1_SFT",
    "sft_num_train_epochs": 1,
    "sft_per_device_train_batch_size": 4,
    "sft_per_device_eval_batch_size": 4,
    "sft_warmup_steps": 5,
    "sft_prediction_loss_only": True,
    "sft_fp16": False # XPU sometimes has issues with fp16 in older versions or specific setups, let's try False or BF16
}

# tokenizer, model 준비
model = AutoModelForCausalLM.from_pretrained(cfg["model_name"]).to(cfg["device"])
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    cfg["model_name"],
    bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>',
    padding_side="right",
    model_max_length=512,
)


step_02_show_info(cfg)


show_base_model_and_dataset(cfg, model, tokenizer)
show_sft_and_rm_dataset(cfg)
run_sft(cfg, model, tokenizer)
test_04()
test_05()