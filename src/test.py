from dotenv import load_dotenv


load_dotenv()
from unsloth import FastLanguageModel
from utils.inference.run_a_message import run_a_message
from utils.inference.run_messages import run_messages
from config.train_config import save_path


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=save_path,  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# run a message here
messages = [
    {
        "role": "user",
        "content": "hello"
    },
    {
        "role": "assistant",
        "content": "hello呀"
    },
    {
        "role": "user",
        "content": "在干嘛呢？"
    },
    {
        "role": "assistant",
        "content": "上班呀\n你呢"
    },
    {
        "role": "user",
        "content": "我也在上班"
    },
    {
        "role": "assistant",
        "content": "你是做什么的？"
    },
    {
        "role": "user",
        "content": "程序员\n不过现在在失业中"
    },
]
# run_a_message(model, tokenizer, messages)

run_messages(model, tokenizer, messages)
