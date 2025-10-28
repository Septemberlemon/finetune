# 训练名称
train_name = "bad_woman_32b"

# 项目根目录
root_path = "/home/u/finetune"

# 数据集文件路径
train_dataset_path = root_path + "/data/bad_woman/train.json"
eval_dataset_path = root_path + "/data/bad_woman/eval.json"

# 数据集格式化信息
conversations_key_name = "conversations"
sharegpt_style = True

# 模型保存路径
save_path = root_path + f"/lora_model/{train_name}"

# 模型基础参数
MODEL_NAME = "unsloth/Qwen3-32B"
CHAT_TEMPLATE = "qwen-3"

MAX_SEQ_LEN = 2048
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False
FULL_FINETUNE = False

# lora模型参数
lora_rank = 32  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
lora_alpha = 32  # Best to choose alpha = rank or rank*2
lora_dropout = 0  # Supports any, but = 0 is optimized
bias = "none"  # Supports any, but = "none" is optimized
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj", ]
use_gradient_checkpointing = "unsloth"  # True or "unsloth" for very long context
random_state = 3407
use_rslora = False  # We support rank stabilized LoRA
loftq_config = None  # And LoftQ

# 训练参数
per_device_train_batch_size = 2
gradient_accumulation_steps = 4  # Use GA to mimic batch size!
warmup_steps = 5
num_train_epochs = 30  # Set this for 1 full training run.
max_steps = 30  # use this if u don't use num_train_epochs
learning_rate = 2e-4  # Reduce to 2e-5 for long training runs
logging_steps = 1
optim = "adamw_8bit"
weight_decay = 0.01
lr_scheduler_type = "linear"
seed = 3407
report_to = "none"  # Use this for WandB etc

# 验证参数
metric_for_best_model = "eval_loss"  # metric we want to early stop on
greater_is_better = False  # the lower the eval loss, the better
load_best_model_at_end = True  # MUST USE for early stopping
eval_strategy = "epoch"
save_strategy = "epoch"  # save model every N steps
output_dir = root_path + "/lora_model/checkpoints"  # location of saved checkpoints for early stopping
save_total_limit = 3  # keep ony 3 saved checkpoints to save disk space
fp16_full_eval = True
per_device_eval_batch_size = 2
eval_accumulation_steps = 4
save_steps = 10  # how many steps until we save the model
eval_steps = 10  # how many steps until we do evaluation
