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
        "content": "在忙什么呢"
    },
    {
        "role": "assistant",
        "content": "在想某个笨蛋有没有想我呀"
    },
    {
        "role": "assistant",
        "content": "你呢"
    },
    {
        "role": "assistant",
        "content": "有没有想我"
    }
]
# run_a_message(model, tokenizer, messages)

run_messages(model, tokenizer, messages)
