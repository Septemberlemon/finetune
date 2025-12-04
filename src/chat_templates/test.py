from jinja2 import Environment, FileSystemLoader


# 模板所在目录（当前目录下）
env = Environment(loader=FileSystemLoader("."))

# 读取外部模板文件
template = env.get_template("inference_chat_template.jinja")

# 定义测试数据集
messages = [
    {
        "role": "user",
        "content": "hello"
    },
    # {
    #     "role": "assistant",
    #     "content": "hello\nworld"
    # },
    # {
    #     "role": "assistant",
    #     "content": "world"
    # },
]

output = template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
print("#" * 60)
print(output)
print("#" * 60)
