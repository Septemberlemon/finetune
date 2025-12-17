from jinja2 import Environment, FileSystemLoader


# 模板所在目录（当前目录下）
env = Environment(loader=FileSystemLoader("."))

# 读取外部模板文件
template = env.get_template("inference_chat_template.jinja")

# 定义测试数据集
messages = [
    {
        "role": "system",
        "content": "you are a helpful assistant"
    },
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
        "content": "我也在上班\n不过"
    },
]

output = template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
print("#" * 60)
print(output, end="$$$")
print("#" * 60)
