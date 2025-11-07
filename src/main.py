from transformers import AutoModelForCausalLM, AutoTokenizer


local_path = "/home/u/.cache/huggingface/hub/models--unsloth--Qwen3-14B/snapshots/b8755c0b498d7b538068383748d6dc20397b4d1f"
tokenizer = AutoTokenizer.from_pretrained(local_path)
tokenizer.apply_chat_template("")
model = AutoModelForCausalLM.from_pretrained(local_path)
model.generate()
