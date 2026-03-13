import os

try:
    root_path = os.path.dirname(os.path.abspath(__file__))
except:
    root_path = os.getcwd()

def step_01_modify_files():
    modifications = [
        {
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
                    "new": "from tqdm import tqdm",
                    "old": "from tqdm.notebook import tqdm"},
            ],
        },
        {
            "file": f"{root_path}/chatgpt/trainer/base.py",
            "changes": [
                {
                    "line": 8,
                    "new": "from tqdm import tqdm",
                    "old": "from tqdm.notebook import tqdm"
                },
            ]
        },
        {
            "file": f"{root_path}/chatgpt/trainer/rm.py",
            "changes": [
                {
                    "line": 8,
                    "new": "from tqdm import tqdm",
                    "old": "from tqdm.notebook import tqdm"
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