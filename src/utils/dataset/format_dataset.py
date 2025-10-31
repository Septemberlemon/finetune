from unsloth.chat_templates import standardize_sharegpt


def format_dataset(dataset, tokenizer, conversations_key_name="conversations", sharegpt_style=False):
    def formatting_prompts_func(examples):
        conversations = examples[conversations_key_name]
        texts = [tokenizer.apply_chat_template(cs, tokenize=False, add_generation_prompt=False) for cs in conversations]
        return {"text": texts, }

    if sharegpt_style:
        dataset = standardize_sharegpt(dataset)
    return dataset.map(formatting_prompts_func, batched=True)
