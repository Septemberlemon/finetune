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
# for i in range(len(formatted_train_dataset)):
#     print(len(tokenizer(formatted_train_dataset[i]["text"])["input_ids"]))
#
# for i in range(len(formatted_eval_dataset)):
#     print(len(tokenizer(formatted_eval_dataset[i]["text"])["input_ids"]))
#
# print(formatted_train_dataset[0]["text"])
# 1. 拿出你那条看起来很完美的数据
raw_text = formatted_train_dataset[0]["text"]

# 2. 让 Tokenizer 把把它吃进去（转成 ID），再吐出来（Decode）
# 注意：这一步模拟了模型真正看到的内容
input_ids = tokenizer(raw_text, add_special_tokens=False)["input_ids"]
decoded_text = tokenizer.decode(input_ids)

# 3. 对比“吃进去前”和“吐出来后”
print('\n' in raw_text)
print('\n' in decoded_text)

print("-" * 20)
print("原始片段:", repr(raw_text[raw_text.find("美甲"):raw_text.find("美甲")+10]))
print("模型片段:", repr(decoded_text[decoded_text.find("美甲"):decoded_text.find("美甲")+10]))
