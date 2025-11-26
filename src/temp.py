# from datasets import load_dataset
# from unsloth.chat_templates import standardize_sharegpt
# from transformers import AutoTokenizer
#
#
# def get_dataset(file_path, split="train"):
#     dataset = load_dataset("json", data_files=file_path, split=split)
#     return dataset
#
#
# def format_dataset(dataset, tokenizer, conversations_key_name="conversations", sharegpt_style=False):
#     def formatting_prompts_func(examples):
#         conversations = examples[conversations_key_name]
#         texts = []
#         for conversation in conversations:
#             split_conversation = []  # 分割后的一个对话历史
#             for message in conversation:
#                 if message["role"] == "assistant":
#                     message_text = message["content"]
#                     split_message_text = message_text.split("\n")
#                     for fragment in split_message_text:
#                         split_conversation.append({"role": "assistant", "content": fragment})
#                 else:
#                     split_conversation.append(message)
#             temp = tokenizer.apply_chat_template(split_conversation, tokenize=False, add_generation_prompt=False)
#             texts.append(temp)
#         return {"text": texts, }
#
#     if sharegpt_style:
#         dataset = standardize_sharegpt(dataset)
#     return dataset.map(formatting_prompts_func, batched=True)
#
#
# eval_dataset_path = "/home/u/finetune/data/bad_woman/eval.json"
# local_path = "/home/u/.cache/huggingface/hub/models--unsloth--qwen3-32b-bnb-4bit/snapshots/7f721e74a6a8cc9ee352f7e49303a2c1705f9083"
#
# tokenizer = AutoTokenizer.from_pretrained(local_path)
# eval_dataset = get_dataset(eval_dataset_path)
# formatted_eval_dataset = format_dataset(eval_dataset, tokenizer, "conversations", True)
# print(formatted_eval_dataset["text"][0])
