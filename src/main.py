from transformers import AutoTokenizer


model_name = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Successfully loaded tokenizer for model: '{model_name}'\n")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# --- 2. 定义我们要测试的词汇列表 ---
words_to_test = [
    "Hausmeister",  # Standard German
    "Plebeschil",  # Dialect
    "今天天气不错",  # Dialect
    "G'schicht",  # Dialect with apostrophe
    "Zwetschgenkrampus",  # Complex compound word
    "nimmamea"  # Dialect
]

# --- 3. 循环遍历每个词，进行分词并评估 ---
print("--- Tokenization Analysis ---")
for word in words_to_test:
    # 使用 .tokenize() 方法来查看词被分解成的字符串列表
    tokens = tokenizer.tokenize(word)

    # 打印结果
    print(f"Word: '{word}'")
    print(f" -> Tokens: {tokens}")
    print(f" -> Number of tokens: {len(tokens)}")
    print("-" * 25)
