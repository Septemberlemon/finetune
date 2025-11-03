## transformers是huggingface生态下的核心库，用于加载、训练和推理主流的Transformer架构模型

使用`pip install transformers`即可安装

***

### pipeline

使用`pipeline`创建一个对话管道，它的作用是快速调用模型进行对话：

```python
from transformers import pipeline

generator = pipeline("text-generation", model="unsloth/Qwen3-14B")
```

上述代码会从**Huggingface Hub**下载模型到本地，并加载模型，得到的`generator`是一个`Pipeline`对象`pipeline`函数的第一个参数是`task`，用于指定具体的人物类型，`model`参数用来指定具体模型名称，对于从远程下载的模型来说，直接指定为官网上的模型名称即可

接下来可以直接调用`generator`本身来进行文本续写：

```python
prompt = "今天天气不错"
results = generator(prompt, max_length=50)
print(results[0]["generated_text"])
```

它可能得到：

```text
今天天气不错，阳光明媚，我打算去公园散步，顺便买点东西。我先去超市买了些水果和蔬菜，然后去公园。公园里人很多，我坐在长椅上休息，看到孩子们在玩耍，感觉
```

`generator`本身接收一个`prompt`参数外加若干关键字参数，`prompt`参数可以是一个**str**或者**list[str]**，对于前者，调用结果将得到一个**list[dict]**、对于后者，调用结果将得到一个**list[list[dict]]**，例如：

```python
prompts = ["今天天气不错，", "法国的首都是"]

results = generator(prompts, max_length=50, num_return_sequences=3)
```

它拿到的`results`是一个长度为**2**的**list**，其中两个元素分别代表对`prompt`中两段文本续写的结果

`results`中的每个元素又是一个长度为**3**的**list**，其中每个元素代表对一段文本进行续写的一种结果，因为指定了`num_return_sequences=3`，所以对一个文本将会有三个生成结果

最内层的是一个**python dict**，代表对**一段文本的一个生成结果**，整个`results`中共有**6**个这样的**dict**

通过：

```python
for result in results:
    for i in range(3):
        print(result[i]["generated_text"])
```

即可打印所有生成结果

对于`prompt`为单个**str**的情况，它得到的结果只是少了最外层的对应不同**prompt**的一层列表

下面介绍一些`Pipeline`对象的其他调用参数：

`max_length`：
