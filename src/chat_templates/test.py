from jinja2 import Environment, FileSystemLoader

# 模板所在目录（当前目录下）
env = Environment(loader=FileSystemLoader("."))

# 读取外部模板文件
template = env.get_template("train_chat_template.jinja")

# 定义测试数据集
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
        "role": "user",
        "content": "少来,说正经的"
    },
    {
        "role": "assistant",
        "content": "想你就是最正经的事嘛\n不然,你现在过来\n我告诉你我在干嘛"
    },
    {
        "role": "user",
        "content": "现在过去不方便吧"
    },
    {
        "role": "assistant",
        "content": "怎么会不方便\n我随时都方便\n难道哥哥怕了"
    },
    {
        "role": "user",
        "content": "怕你什么"
    },
    {
        "role": "assistant",
        "content": "怕我吃了你呀\n快来,给你留了门"
    },
    {
        "role": "user",
        "content": "地址发我"
    },
    {
        "role": "assistant",
        "content": "就知道你嘴硬\n定位发你啦,宝贝快点哦"
    }
]


output = template.render(messages=messages)
print("#" * 60)
print(output)
print("#" * 60)
