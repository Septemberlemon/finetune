from transformers import TextStreamer
# 下面两个import是为了应对Linux终端的中文Backspace异常问题
import readline
import rlcompleter


def run_messages(model, tokenizer, dialogue_history=None):
    print("--- 对话开始 ---")
    print("输入 '/quit' 来结束对话。")
    if dialogue_history is None:
        dialogue_history = []
    else:
        for message in dialogue_history:
            print(message["role"] + "：")
            print(message["content"])
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    while True:
        if dialogue_history[-1]["role"] != "user":
            # 接收用户输入
            user_input = input("user：\n")

            # 检查退出条件
            if user_input == "/quit":
                print("--- 对话结束 ---")
                break

            # 将用户输入追加到对话历史中
            dialogue_history.append({"role": "user", "content": user_input})

        # 格式化完整的对话历史，为模型生成做准备
        text = tokenizer.apply_chat_template(
            dialogue_history,
            tokenize=False,
            add_generation_prompt=True,  # 关键！为 assistant 的回答做引导
            enable_thinking=False,
        )

        # 将格式化后的文本转换成模型输入
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        # --- 开始生成 ---
        print("assistant：")

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7, top_p=0.8, top_k=20,
            streamer=streamer,  # 使用 streamer 来实时打印
        )

        # 模型生成到<|im_end|>后立即停止，不会输出后续的"\n"，所以最后一个token是<|im_end|>
        assistant_response_ids = outputs[:, inputs.input_ids.shape[1]:-1]
        assistant_response = tokenizer.batch_decode(assistant_response_ids, skip_special_tokens=False)[0]

        # 将模型回答追加到对话历史中
        dialogue_history.append({"role": "assistant", "content": assistant_response})
