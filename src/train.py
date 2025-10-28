from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from config.train_config import *
from utils.dataset.get_dataset import get_dataset
from utils.dataset.format_dataset import format_dataset


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,  # Context length - can be longer, but uses more memory
    load_in_4bit=LOAD_IN_4BIT,  # 4bit uses much less memory
    load_in_8bit=LOAD_IN_8BIT,  # A bit more accurate, uses 2x memory
    full_finetuning=FULL_FINETUNE,  # We have full finetuning now!
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template=CHAT_TEMPLATE,  # change this to the right chat_template name
)

train_dataset = get_dataset(train_dataset_path)
eval_dataset = get_dataset(eval_dataset_path)
formatted_train_dataset = format_dataset(train_dataset, tokenizer, conversations_key_name, sharegpt_style)
formatted_eval_dataset = format_dataset(eval_dataset, tokenizer, conversations_key_name, sharegpt_style)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=target_modules,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias=bias,
    use_gradient_checkpointing=use_gradient_checkpointing,
    random_state=random_state,
    use_rslora=use_rslora,
    loftq_config=loftq_config,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train_dataset,
    eval_dataset=formatted_eval_dataset,  # Can set up evaluation!
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        # max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        optim=optim,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
        report_to=report_to,
        # for early stop below
        fp16_full_eval=fp16_full_eval,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_accumulation_steps=eval_accumulation_steps,
        output_dir=output_dir,  # location of saved checkpoints for early stopping
        save_strategy=save_strategy,  # save model every N steps
        # save_steps=save_steps,  # how many steps until we save the model
        save_total_limit=save_total_limit,  # keep ony 3 saved checkpoints to save disk space
        eval_strategy=eval_strategy,  # evaluate every N steps
        # eval_steps=eval_steps,  # how many steps until we do evaluation
        load_best_model_at_end=load_best_model_at_end,  # MUST USE for early stopping
        metric_for_best_model=metric_for_best_model,  # metric we want to early stop on
        greater_is_better=greater_is_better,  # the lower the eval loss, the better
    ),
)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,  # How many steps we will wait if the eval loss doesn't decrease
    # For example the loss might increase, but decrease after 3 steps
    early_stopping_threshold=0.0,  # Can set higher - sets how much loss should decrease by until
    # we consider early stopping. For eg 0.01 means if loss was
    # 0.02 then 0.01, we consider to early stop the run.
)


if __name__ == "__main__":
    trainer.add_callback(early_stopping_callback)
    trainer_stats = trainer.train()

    model.save_pretrained(save_path)  # Local saving
    tokenizer.save_pretrained(save_path)
