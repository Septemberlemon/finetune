from utils.dataset.get_dataset import get_dataset
from utils.dataset.format_dataset import format_dataset
from config.train_config import *
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,  # Context length - can be longer, but uses more memory
    load_in_4bit=LOAD_IN_4BIT,  # 4bit uses much less memory
    load_in_8bit=LOAD_IN_8BIT,  # A bit more accurate, uses 2x memory
    full_finetuning=FULL_FINETUNE,  # We have full finetuning now!
)

# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template=CHAT_TEMPLATE,  # change this to the right chat_template name
# )
train_dataset = get_dataset(train_dataset_path)
eval_dataset = get_dataset(eval_dataset_path)
formatted_train_dataset = format_dataset(train_dataset, tokenizer, conversations_key_name, sharegpt_style)
formatted_eval_dataset = format_dataset(eval_dataset, tokenizer, conversations_key_name, sharegpt_style)
print(formatted_train_dataset[0]["text"])
