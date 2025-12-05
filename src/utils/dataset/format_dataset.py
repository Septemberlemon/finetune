from unsloth.chat_templates import standardize_sharegpt


def format_dataset(dataset, tokenizer, conversations_key_name="conversations", sharegpt_style=False):
    def formatting_prompts_func(examples):
        conversations = examples[conversations_key_name]
        texts = []
        for conversation in conversations:
            for message in conversation:
                if message["role"] == "assistant":
                    message["content"] += "\n"
            temp = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            texts.append(temp)
        return {"text": texts}

    if sharegpt_style:
        dataset = standardize_sharegpt(dataset)
    return dataset.map(formatting_prompts_func, batched=True)


# def format_dataset(dataset, tokenizer, conversations_key_name="conversations", sharegpt_style=False):
#     def formatting_prompts_func(examples):
#         conversations = examples[conversations_key_name]
#         texts = [tokenizer.apply_chat_template(cs, tokenize=False, add_generation_prompt=False) for cs in conversations]
#         return {"text": texts}
#
#     if sharegpt_style:
#         dataset = standardize_sharegpt(dataset)
#     return dataset.map(formatting_prompts_func, batched=True)

# def format_dataset(dataset, tokenizer, conversations_key_name="conversations", sharegpt_style=False):
#     def formatting_prompts_func(examples):
#         conversations = examples[conversations_key_name]
#         texts = []
#         for conversation in conversations:
#             modified_conversation = []  # 插入 <|im_end|> 后的一个对话历史
#             for message in conversation:
#                 if message["role"] == "assistant":
#                     split_messages = [string.strip() for string in message["content"].split("\n")]
#                     modified_conversation.append({"role": "assistant", "content": "<|im_end|>".join(split_messages)})
#                 else:
#                     modified_conversation.append(message)
#             temp = tokenizer.apply_chat_template(modified_conversation, tokenize=False, add_generation_prompt=False)
#             texts.append(temp)
#         return {"text": texts}
#
#     if sharegpt_style:
#         dataset = standardize_sharegpt(dataset)
#     return dataset.map(formatting_prompts_func, batched=True)

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
#                     for i, fragment in enumerate(split_message_text):
#                         # if i == len(split_message_text) - 1:
#                         #     split_conversation.append({"role": "assistant", "content": fragment.strip() + "\n"})
#                         # else:
#                         split_conversation.append({"role": "assistant", "content": fragment.strip()})
#                 else:
#                     split_conversation.append(message)
#             temp = tokenizer.apply_chat_template(split_conversation, tokenize=False, add_generation_prompt=False)
#             texts.append(temp)
#         return {"text": texts}
#
#     if sharegpt_style:
#         dataset = standardize_sharegpt(dataset)
#     return dataset.map(formatting_prompts_func, batched=True)
