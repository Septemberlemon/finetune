# transformers是huggingface生态下的核心库，用于加载、训练和推理主流的Transformer架构模型

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

`results`中的每个元素又是一个长度为**3**的**list**，其中每个元素代表对一段文本进行续写的一种结果，这是因为指定了`num_return_sequences=3`，所以对一个文本将会有三个生成结果

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

`max_length`：它决定了文本的最大长度，例如传进去的文本长度为**20**个**token**，`max_length`为**50**，则会将文本续写到**50**个**token**，如果没到**50**个**token**遇到中止**token**也将停止续写

`max_new_tokens`：限定新生成的**token**的最大数量

`temperature`：温度，用于控制模型倾向于**更保守**还是**更具创造性**

`top_k`：限定输出时从概率最高的**k**个**token**进行采样

`top_p`：输出采样时从概率最高的**token**开始，从高到低累加**token**的概率，直到总和达到 **p**，到达**p**后将被选中的**token**们将被用于采样

如果同时指定了`top_k`和`top_p`，则会先使用`top_k`过滤前**k**个**token**，再使用概率**p**进行过滤，若累加不到**p**，则取全部候选**token**进行采样

#### 从本地文件中加载模型

将`model`参数指定为本地文件路径即可，例如一般从**Hugggingface Hub**下载的模型位于：`~/.cache/huggingface/hub`下，对于上述示例中的模型，将其下载到本地后可以通过：

```python
generator = pipeline("text-generation", model="/home/u/.cache/huggingface/hub/models--unsloth--Qwen3-14B/snapshots/b8755c0b498d7b538068383748d6dc20397b4d1f")
```

进行加载，**注意**路径一定要指定到**snapshot**再下面一层

***

## AutoTokenizer、AutoModelForCausalLM

这是另一种加载模型的办法，它将**分词器**和**模型**分别加载，`pipeline`仅用于推理使用，使用`AutoTokenizer`和`AutoModelForCausalLM`能用于训练和推理

首先导入二者：

```python
from transfomers import AutoTokenizer, AutoModelForCausalLM
```

现代**llm**发布时往往将**分词器**和**模型**合并发布，这意味着往往可以使用同一个`repo_id`或者**本地路径**进行二者的加载，如：

根据`repo_id`从**Huaggingface Hub**拉取分词器和模型（有缓存会自动使用）：

```python
repo_id = "unsloth/Qwen3-14B"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id)
```

从本地路径加载：

```python
local_path = "/home/u/.cache/huggingface/hub/models--unsloth--Qwen3-14B/snapshots/b8755c0b498d7b538068383748d6dc20397b4d1f"

tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForCausalLM.from_pretrained(local_path)
```

优先级为先检查传入的参数是否为本地文件夹，再检查**Hub**是否有对应仓库

***

### 关于tokenizer

从字符串到**token_id**列表的过程往往需要经历下述过程：

**Normalizer → PreTokenizer → Model (BPE merges) → PostProcessor**

* **Normalizer**部分会对字符串做一些预处理，它接收原始字符串，输出预处理后的字符串

* **Pretokenizer**接收处理后的字符串，例如`"今天天气真好,I wanna go swimming"`，按照一定的规则将字符串拆为若干块子字符串（例如按照空格分割），拿到类似于` ['今天', '天气', '真', '好', ',I', ' wanna', ' go', ' swimming']`这样的切分，接着它会做一个映射，将每个子字符串转为字节流，再将每个字节流按照一个字节对应一个**可打印字符**的映射关系做映射，得到`['ä»Ĭå¤©å¤©æ°ĶçľŁå¥½', ',I', 'Ġwanna', 'Ġgo', 'Ġswimming']`作为输出，这被称为**pre-token**，具体字节到**可打印字符**的映射关系见后文，此外它输出时还会带上**offset_mapping**作为记录每个子字符串对应原字符串中的索引位置的信息，实际输出为：

    `[('ä»Ĭå¤©å¤©æ°ĶçľŁå¥½', (0, 6)), (',I', (6, 8)), ('Ġwanna', (8, 14)), ('Ġgo', (14, 17)), ('Ġswimming', (17, 26))]`

    上述输出使用代码：`print(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("今天天气真好,I wanna go swimming"))`得到

* **Model(BPE merges)**是实际负责分词的模型部分，它拿到上一步输出的**pre-tokens**和**offset_mapping**，在每个**pre-token**内部使用**BPE**算法进行进一步的拆分（或者说合并，这取决于看待每个**pre-token**的方式），拿到最终的**tokens**（该算法更关心频率而非语意相关性）：`['ä»Ĭå¤©','å¤©æ°Ķ','çľŁ','å¥1⁄2','ï1⁄4Į','I','Ġwanna','Ġgo','Ġswimming']`

    使用代码：

    `print(tokenizer.tokenize("今天天气真好,I wanna go swimming"))`

    拿到输出的**tokens**：

    `['ä»Ĭå¤©', 'å¤©æ°Ķ', 'çľŁ', 'å¥½', ',I', 'Ġwanna', 'Ġgo', 'Ġswimming']`

    最后它会将每个**token**对应到**token_id**，得到最终输出：

    `[100644, 104307, 88051, 52801, 37768, 32733, 728, 23380]`

    使用代码：

    `print(tokenizer("今天天气真好,I wanna go swimming")["input_ids"])`可以拿到上述**token_ids**输出

    并且它也会对**offset_mapping**进行处理，使用下述代码：

    `print(tokenizer("今天天气真好,I wanna go swimming", return_attention_mask=False, return_offsets_mapping=True))`

    拿到输出：

    ```
    {
    	'input_ids': [100644, 104307, 88051, 52801, 37768, 32733, 728, 23380], 
    	'offset_mapping': [(0, 2), (2, 4), (4, 5), (5, 6), (6, 8), (8, 14), (14, 17), (17, 26)]
    }
    ```

* **PostProcessor**会做一些后处理，它是和**Normalizer**一样是非必要的

上述介绍的是在字节层面进行的分词，实际上也有很多其他方法，但**byte-level**是目前最主流的方法，因为在**BPE**阶段按照词表进行合并，且未合并的字符保持为单个字符，而单个字符也在词表中，它完全的避免了**oov**。

拿到分词器对字符串编码后的**token_id**列表后，将其喂给真正负责语言逻辑处理的模型，拿到输出的**token_id**再调用**tokenizer**进行解码，解码部分为将**token_id**转为**token**，再按照特定映射（下文有映射表）映射回字节流，再将字节流转为人类可读的字符串

#### 正向编码

##### 直接调用

通过直接调用`tokenizer`对象，即可得到分词后的结果，如：

```python
prompt = "今天天气真好"
encoded_input = tokenizer(prompt)
print(encoded_input)
```

这将得到一个`transformers.tokenization_utils_base.BatchEncoding`对象：

```shell
{'input_ids': [100644, 104307, 88051, 52801], 'attention_mask': [1, 1, 1, 1]}
```

它是一个类字典对象，其中`"input_ids"`即为分词后的结果，它代表每个**token**对应的**token_id**构成的列表，实际上它也真是一个**list[int]**对象；`"attention_mask"`是一个和`"input_ids"`等长的**list[int]**，代表注意力掩码，这部分后续介绍

也可以批量处理一批字符串，如：

```python
prompts = ["今天天气真好", "法国的首都是巴黎"]
encoded_input = tokenizer(prompts)
print(encoded_input)
```

这也将得到一个`transformers.tokenization_utils_base.BatchEncoding`对象，键仍然是`"input_ids"`和`"attention_mask"`：

```shell
{
	'input_ids': [[100644, 104307, 88051, 52801], [104328, 9370, 59975, 100132, 106004]],
	'attention_mask': [[1, 1, 1, 1], [1, 1, 1, 1, 1]]
}
```

唯一不同的是多嵌套了一层列表

下面介绍一些直接调用`tokenizer`的参数：

`max_length`：用于指定文本的最大**token**数量

`truncation`：指定其为`True`后会将编码后**token**数量大于`max_length`的**token**序列进行截断，取前`max_length`个**token**

`padding`：指定填充，设定为`True`后会将编码后**token**数量小于特定值的**token**序列进行填充，补充长度到特定值，这里的特定值**所有token序列长度中的最大值**；设定为`"max_length"`后会将长度填充为`"max_length"`；填充的**token**为一个特殊的专门用于填充的**token**，可以通过`tokenizer.all_special_tokens`查看所有特殊**token**，其中就包含用于填充的**token**对应的字符串，它往往是一个包含**“pad“**的字符串；默认在**token**序列的左边进行填充，可以通过指定`tokenizer.padding_side = "right"`指定在序列右边进行填充，默认使用左填充是因为新生成的**token**可以直接在后面拼接，使用右填充若是直接在后面拼接将会导致新生成的**token**和输入**token**序列中间包含若干填充**token**

`return_offsets_mapping`：指定此参数为`True`后，将会在返回的对象中增添一个键值对，键为`"offsets_mapping"`，值为一个**list[tuple[int]]**或者**list[list[tuple[int]]]**（这取决于传进去的是字符串还是字符串列表），代表分词切分位置对应的原字符串中的位置索引，如：

```shell
'offset_mapping': [(0, 2), (2, 4), (4, 5), (5, 6)]
'offset_mapping': [[(0, 2), (2, 4), (4, 5), (5, 6)], [(0, 2), (2, 3), (3, 4), (4, 6), (6, 8)]]
```

`return_tensors`：用于指定返回的`"input_ids"`和`"attention_mask"`的类型，它们默认是嵌套列表，可以通过指定此参数为`pt`、`tf`、`np`分别指定返回的对象类型为`torch.tensor`、`tensorflow.tensor`、`numpy.ndarray`，注意，如果没有指定`truncation`或者`padding`，则可能会因为句子编码后的**token**序列的长度不同而无法将嵌套列表转为张量，例如：

```python
prompts = ["今天天气真好", "法国的首都是巴黎"]
encoded_input = tokenizer(prompts, return_tensors="pt")
print(encoded_input)
```

将得到报错：

```text
ValueError: Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (`input_ids` in this case) have excessive nesting (inputs type `list` where type `int` is expected).
```

所以应该通过指定其他参数保证编码后**token**序列长度一致，才能使`return_tensors`正常生效，如：

```python
prompts = ["今天天气真好", "法国的首都是巴黎"]
encoded_input = tokenizer(prompts, padding=True, return_tensors="pt")
print(encoded_input)
```

这将得到：

```shell
{
	'input_ids': 
		tensor([[151654, 100644, 104307,  88051,  52801],
			    [104328,   9370,  59975, 100132, 106004]]),
	'attention_mask': 
		tensor([[0, 1, 1, 1, 1],
        	   [1, 1, 1, 1, 1]])
}
```

此外如果返回的除了这二者还有其他键值对（如前面的**offsets_mapping**），其也会被一并转换为相应类型

##### tokenizer.tokenize方法

此方法与直接调用不同，它将直接返回输入字符串切分后的列表，如：

```python
prompt = "今天天气真好"
print(tokenizer.tokenize(prompt))
```

这将得到：

```shell
['ä»Ĭå¤©', 'å¤©æ°Ķ', 'çľŁ', 'å¥½']
```

下面是单字节到**便于打印**字符的映射：

```python
def bytes_to_unicode():
    """
    生成字节到Unicode字符的正向映射表
    返回字典：{byte_value: unicode_char}
    """
    # 原始保留的字节范围
    bs = (
        list(range(ord("!"), ord("~") + 1)) +          # ASCII可打印字符（33-126）
        list(range(ord("¡"), ord("¬") + 1)) +          # 西班牙语特殊字符（161-172）
        list(range(ord("®"), ord("ÿ") + 1))            # 其他扩展字符（174-255）
    )
    
    cs = bs.copy()  # 初始字符列表
    n = 0
    
    # 遍历所有可能的字节（0-255）
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)  # 超出原始范围的字节映射到更高Unicode码位
            n += 1
    
    # 将码位转换为Unicode字符
    cs = [chr(code) for code in cs]
    
    return dict(zip(bs, cs))

print(sorted(bytes_to_unicode().items()))
```

```shell
[(0, 'Ā'), (1, 'ā'), (2, 'Ă'), (3, 'ă'), (4, 'Ą'), (5, 'ą'), (6, 'Ć'), (7, 'ć'), (8, 'Ĉ'), (9, 'ĉ'), (10, 'Ċ'), (11, 'ċ'), (12, 'Č'), (13, 'č'), (14, 'Ď'), (15, 'ď'), (16, 'Đ'), (17, 'đ'), (18, 'Ē'), (19, 'ē'), (20, 'Ĕ'), (21, 'ĕ'), (22, 'Ė'), (23, 'ė'), (24, 'Ę'), (25, 'ę'), (26, 'Ě'), (27, 'ě'), (28, 'Ĝ'), (29, 'ĝ'), (30, 'Ğ'), (31, 'ğ'), (32, 'Ġ'), (33, '!'), (34, '"'), (35, '#'), (36, '$'), (37, '%'), (38, '&'), (39, "'"), (40, '('), (41, ')'), (42, '*'), (43, '+'), (44, ','), (45, '-'), (46, '.'), (47, '/'), (48, '0'), (49, '1'), (50, '2'), (51, '3'), (52, '4'), (53, '5'), (54, '6'), (55, '7'), (56, '8'), (57, '9'), (58, ':'), (59, ';'), (60, '<'), (61, '='), (62, '>'), (63, '?'), (64, '@'), (65, 'A'), (66, 'B'), (67, 'C'), (68, 'D'), (69, 'E'), (70, 'F'), (71, 'G'), (72, 'H'), (73, 'I'), (74, 'J'), (75, 'K'), (76, 'L'), (77, 'M'), (78, 'N'), (79, 'O'), (80, 'P'), (81, 'Q'), (82, 'R'), (83, 'S'), (84, 'T'), (85, 'U'), (86, 'V'), (87, 'W'), (88, 'X'), (89, 'Y'), (90, 'Z'), (91, '['), (92, '\\'), (93, ']'), (94, '^'), (95, '_'), (96, '`'), (97, 'a'), (98, 'b'), (99, 'c'), (100, 'd'), (101, 'e'), (102, 'f'), (103, 'g'), (104, 'h'), (105, 'i'), (106, 'j'), (107, 'k'), (108, 'l'), (109, 'm'), (110, 'n'), (111, 'o'), (112, 'p'), (113, 'q'), (114, 'r'), (115, 's'), (116, 't'), (117, 'u'), (118, 'v'), (119, 'w'), (120, 'x'), (121, 'y'), (122, 'z'), (123, '{'), (124, '|'), (125, '}'), (126, '~'), (127, 'ġ'), (128, 'Ģ'), (129, 'ģ'), (130, 'Ĥ'), (131, 'ĥ'), (132, 'Ħ'), (133, 'ħ'), (134, 'Ĩ'), (135, 'ĩ'), (136, 'Ī'), (137, 'ī'), (138, 'Ĭ'), (139, 'ĭ'), (140, 'Į'), (141, 'į'), (142, 'İ'), (143, 'ı'), (144, 'Ĳ'), (145, 'ĳ'), (146, 'Ĵ'), (147, 'ĵ'), (148, 'Ķ'), (149, 'ķ'), (150, 'ĸ'), (151, 'Ĺ'), (152, 'ĺ'), (153, 'Ļ'), (154, 'ļ'), (155, 'Ľ'), (156, 'ľ'), (157, 'Ŀ'), (158, 'ŀ'), (159, 'Ł'), (160, 'ł'), (161, '¡'), (162, '¢'), (163, '£'), (164, '¤'), (165, '¥'), (166, '¦'), (167, '§'), (168, '¨'), (169, '©'), (170, 'ª'), (171, '«'), (172, '¬'), (173, 'Ń'), (174, '®'), (175, '¯'), (176, '°'), (177, '±'), (178, '²'), (179, '³'), (180, '´'), (181, 'µ'), (182, '¶'), (183, '·'), (184, '¸'), (185, '¹'), (186, 'º'), (187, '»'), (188, '¼'), (189, '½'), (190, '¾'), (191, '¿'), (192, 'À'), (193, 'Á'), (194, 'Â'), (195, 'Ã'), (196, 'Ä'), (197, 'Å'), (198, 'Æ'), (199, 'Ç'), (200, 'È'), (201, 'É'), (202, 'Ê'), (203, 'Ë'), (204, 'Ì'), (205, 'Í'), (206, 'Î'), (207, 'Ï'), (208, 'Ð'), (209, 'Ñ'), (210, 'Ò'), (211, 'Ó'), (212, 'Ô'), (213, 'Õ'), (214, 'Ö'), (215, '×'), (216, 'Ø'), (217, 'Ù'), (218, 'Ú'), (219, 'Û'), (220, 'Ü'), (221, 'Ý'), (222, 'Þ'), (223, 'ß'), (224, 'à'), (225, 'á'), (226, 'â'), (227, 'ã'), (228, 'ä'), (229, 'å'), (230, 'æ'), (231, 'ç'), (232, 'è'), (233, 'é'), (234, 'ê'), (235, 'ë'), (236, 'ì'), (237, 'í'), (238, 'î'), (239, 'ï'), (240, 'ð'), (241, 'ñ'), (242, 'ò'), (243, 'ó'), (244, 'ô'), (245, 'õ'), (246, 'ö'), (247, '÷'), (248, 'ø'), (249, 'ù'), (250, 'ú'), (251, 'û'), (252, 'ü'), (253, 'ý'), (254, 'þ'), (255, 'ÿ')]
```

这种映射不会影响英文字母的打印，但会影响非英文字符的打印，可通过下述代码查看原始文本：

```python
map_dict = {v: k for k, v in bytes_to_unicode().items()}
texts = ['ä»Ĭå¤©', 'å¤©æ°Ķ', 'çľŁ', 'å¥½']
original_texts = [bytes([map_dict[char] for char in text]).decode("utf-8") for text in texts]
print(original_texts)
```

```shell
['今天', '天气', '真', '好']
```

如果给`tokenize`方法传的是一个字符串列表，它将会把列表中的所有字符串拼接成一个字符串后处理

##### tokenizer.convert_tokens_to_ids方法

此方法用于将**token**列表转换为**id**列表，如：

```python
tokens = ['ä»Ĭå¤©', 'å¤©æ°Ķ', 'çľŁ', 'å¥½']
print(tokenizer.convert_tokens_to_ids(tokens))
```

这将得到：

```shell
[100644, 104307, 88051, 52801]
```

它也能接收单个字符串作为输入，输出将为单个**int**，如：

```python
print(tokenizer.convert_tokens_to_ids("<|vision_pad|>"))
```

会得到：

```text
151654
```

#### 反向解码

##### tokenizer.convert_ids_to_tokens方法

此方法接收一个代表**token_id**序列的整数列表，将其解码为**token**列表，如：

```python
prompt = "今天天气真好"
input_ids = tokenizer(prompt)["input_ids"]
print(tokenizer.convert_ids_to_tokens(input_ids))
```

即可得到：

```shell
['ä»Ĭå¤©', 'å¤©æ°Ķ', 'çľŁ', 'å¥½']
```

它也能接收单个**int**作为输入，输出单个字符串，如：

```python
print(tokenizer.convert_ids_to_tokens(151654))
```

会得到：

```text
<|vision_pad|>
```

##### tokenizer.decode方法

此方法接收一个代表**token_id**序列的、类型为**list[int]**、**np.ndarray**、**torch.tensor**等（基本上直接拿**tokenizer**对字符串直接作用返回的`input_ids`就行了）的对象，将其解码为**token**列表并合并为一个**能正常显示非英文字符**字符串，如：

```python
prompt = "今天天气真好"
input_ids = tokenizer(prompt)["input_ids"]
print(tokenizer.decode(input_ids))
```

即可得到`prompt`本身：

```shell
今天天气真好
```

##### tokenizer.batch_decode方法

此方法接收一个代表**token_ids**序列的、**二级嵌套的**、类型为**list[int]**、**np.ndarray**、**torch.tensor**等（基本上直接拿**tokenizer**对字符串列表直接作用返回的`input_ids`就行了）的对象，将其解码为**token**并各自合并为**能正常显示非英文字符**的字符串，再放进一个列表中返回，如：

```python
prompts = ["今天天气真好", "法国的首都是巴黎"]
print(tokenizer.batch_decode(tokenizer(prompts)["input_ids"]))
```

将得到：

```shell
['今天天气真好', '法国的首都是巴黎']
```

此外如果对一个一级列表作用，它将得到能正常显示非英文字符的分词结果：

```python
prompt = "今天天气真好"
tokenizer.batch_decode(tokenizer(prompt)["input_ids"])
```

这将得到：

```shell
['今天', '天气', '真', '好']
```

这是因为`batch_decode`内部对传进去的列表的每一项分别做`decode`在合并进一个列表返回的原因，而对一个一级列表来说，其内部的每一个`input_id`做`decode`的结果就是其对应的**能正常显示的非英文字符串**，在放进列表中返回就得到了能正常显示的分词结果，也可以自己用`decode`处理：

```python
prompt = "今天天气真好"
input_ids = tokenizer(prompt)["input_ids"]
print([tokenizer.decode(input_id) for input_id in input_ids])
```

这也将得到**能正常显示非英文字符**的分词结果：

```shell
['今天', '天气', '真', '好']
```

此外也可以通过前面介绍过的`return_offsets_mapping`参数拿到返回的**offset_mapping**，然后据其从原字符串中手动切分出**能正常显示非英文字符**的分词结果

对于批量的字符串分词，要正确显示非英文字符，采用上述三种方法迭代处理即可

#### 保存分词器

使用`save_pretrained`方法进行保存，参数填路径即可，如：

```python
tokenizer.save_pretrained("tokenizer")
```

即会将分词器保存到当前脚本所属路径下的`tokenizer`文件夹下

#### apply_chat_template方法

聊天模板是对话类语言模型的分词器带有的方法，用于将一个包含多轮对话的列表转换为预设的特定格式的字符串，**这个接口在数据处理阶段非常常用**

首先使用`tokenizer.chat_template`查看其聊天模板，若没有此属性通常表明此分词器不是对话类语言模型的分词器，无法使用`apply_chat_template`方法

`tokenizer.chat_template`的输出是一个字符串，它采用**Jinja2**模板语言定义了对话的格式化规则，形如：

```jinja2
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n'}}
    {% elif message['role'] == 'assistant' %}
        {{'<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n'}}
    {% else %}
        {{'<|im_start|>system\n' + message['content'] + '<|im_end|>\n'}}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}
```

**jinja2**是一种模板语言，也是一种模板引擎，它的语法非常直观，这里用于定义聊天模板的格式

使用`apply_chat_template`方法，它接收一个**list[dict[str, str]]**，将其按照**chat_template**中定义的模板进行转为单个字符串输出。它也可以接收一个**list[list[dict[str, str]]]**，返回对其中每个**list[dict[str, str]]**分别处理后的字符串构成的列表，如：

```python
conversations = [{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '你好，有什么可以帮您的？'}]
formatted_string = tokenizer.apply_chat_template(conversations, tokenize=False)
print(formatted_string)
```

这将得到一个字符串：

```text
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
<think>

</think>

你好，有什么可以帮您的？<|im_end|>

```

这就是按照聊天模板格式化后的字符串，第一个参数一定要指定为要输入的列表，或者用`conversation`参数名指定

`tokenize`参数默认为`True`，当其为`True`时，其输出将为一个**list[int]**：

```text
[151644, 872, 198, 108386, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271, 108386, 3837, 104139, 73670, 99663, 101214, 11319, 151645, 198]
```

它实际上对应的是上述字符串被**tokenizer**作用后的结果，即：

`tokenizer(tokenizer.apply_chat_template(conversations, tokenize=False))["input_ids"]`

通过解码可以查看其对应分词结果：

```python
print(tokenizer.batch_decode(tokenizer.apply_chat_template(conversations)))
```

```text
['<|im_start|>', 'user', '\n', '你好', '<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n', '<think>', '\n\n', '</think>', '\n\n', '你好', '，', '有什么', '可以', '帮', '您的', '？', '<|im_end|>', '\n']
```

`return_tensors`：这是`apply_chat_template`的一个重要参数，用于指定返回的**token_ids**的类型，类似于前面介绍过的直接调用`tokenizer`的同名参数，一般要喂给模型，指定其为`"pt"`即可。指定该参数为`"pt"`之后，即使`conversations`是一个**list[dict[str, str]]**而非**list[list[dict[str, str]]]**，其也将返回一个二维张量（**shape**为`torch.Size([1, n])`），这是因为后续的`model.generate`拒绝处理一维张量，需要在外层**unsqueeze**一层**batch_size**维度

**除了上述的参数，此方法还有另外一种参数，它们会被传给chat_template用于具体的渲染，这意味着不同的chat_template将会能处理不同的参数，要依据其内容具体而定**，举例来说，**unsloth/Qwen3-14B**的**chat_template**中有这么一段内容：

```jinja2
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
```

里面用到了`add_generation_prompt`和`enable_thinking`两个参数，并指明了具体的使用逻辑，这意味着你将可以给`apply_chat_template`方法传递这两个参数：

`add_generation_prompt`：该参数的默认值为**False**（它不是可变关键字参数，所以有默认值，但是它也能在**chat_template**被用到），根据上述**chat_template**，指定其为**True**之后，将会在返回的**token_ids**后面添加一些**tokens_ids**，它们对应着所谓的**”generation_prompt“**，具体内容取决于上述**chat_template**，举例来说：

```python
token_ids_with_generation_prompt = tokenizer.apply_chat_template(conversations, add_generation_prompt=True)
print(tokenizer.batch_decode(token_ids_with_generation_prompt))
```

这将得到：

```text
['<|im_start|>', 'user', '\n', '你好', '<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n', '<think>', '\n\n', '</think>', '\n\n', '你好', '，', '有什么', '可以', '帮', '您的', '？', '<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n']
```

可以看到末尾多了一些**token**：`'<|im_start|>', 'assistant', '\n'`，这就是所谓的**generation_prompt**，它用于提示模型该作为助手回答前文的问题，而不是普通的续写。

`enable_thinking`：根据上述**chat_template**，该参数必须在`add_generation_prompt`指定为**True**且其自身指定为**False**时才会产生影响，它将在后面加上字符串`"<think>\n\n</think>\n\n"`对应的**token_ids**，这将使得模型接下来处理的时候跳过思考阶段，这是一种直观的方法：

```python
token_ids_with_generation_prompt_and_empty_thinking = tokenizer.apply_chat_template(conversations, add_generation_prompt=True)
print(tokenizer.batch_decode(token_ids_with_generation_prompt))
```

```text
['<|im_start|>', 'user', '\n', '你好', '<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n', '<think>', '\n\n', '</think>', '\n\n', '你好', '，', '有什么', '可以', '帮', '您的', '？', '<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n', '<think>', '\n\n', '</think>', '\n\n']
```

这些**chat_template**内部会用到的参数和`tokenize`、`return_tensors`作用是互相独立的

***

### 关于model

这是真正负责处理语言逻辑的部分，在**tokenizer**对字符串处理后要将**token_ids**交由模型进行续写，模型输出**token_id**并返回给**tokenizer**进行解码

前面介绍了使用`AutoModelForCausalLM.from_pretrained`方法加载模型的方法，下面介绍该方法的一些参数：

`device_map`：此参数用于指定将模型的各部分分别加载到哪个设备上，不显式指明该参数，模型将会被全部加载到内存中，其**device**为**cpu**。指定该参数为**”auto"**，库将自动检查可用硬件资源，优先加载到显存中并指定**device**为**cuda**，后续显存不足将会把其余部分**offload**到内存中

`dtype`：此参数用于指定模型的参数的数据类型，**默认为`torch.float32`**。指定其为**auto**后将会根据模型配置文件中指定的数据类型进行加载。加载后可使用`print(model.dtype)`查看其值

建议将二者都指定为**auto**

***

#### generate方法

此方法用于生成，类似于本文档开头介绍过的直接调用`Pipeline`，它接收的首个参数为`"iputs"`，是一个二维的张量，视第一个维度为**batch_size**，视第二个维度为具体的**token_id**序列，其输出一个二维张量，第一个维度保持不变，在第二个维度上进行拓展，返回续写后的**token_id**序列，如：

```python
model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(model.device)
generated_ids = model.generate(model_inputs["input_ids"], max_new_tokens=50)
print(tokenizer.batch_decode(generated_ids))
```

这将得到：

```text
["A list of colors: red, blue, green, yellow, black, white, orange, purple, pink, brown, gray, teal, magenta, cyan, indigo, violet, gold, silver.  How many colors are in the list? Let's count them one by"]
```

如果不指定`max_new_tokens`进行续写长度的限制，其将使用模型配置文件`generation_config.json`中指定的`max_length`进行限定，而`max_length`可能非常大，所以一般需要手动指定，另外也可以通过参数`max_length`进行限定。在到达上限之前，它可能因为生成了终止**token**而停止续写

此外它也支持`top_p`、`top_k`、`temperature`三个参数，具体作用已经在本文档开头的**pipeline**部分介绍过

此外，它还支持`attention_mask`参数，因此你可以采用这种写法：

```python
model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, max_new_tokens=50)
print(tokenizer.batch_decode(generated_ids))
```

注意力掩码在传递单个序列时全为**1**，在传递批量数据时，因为各序列长短不一，要进行**padding**（在前文直接调用`tokenizer`部分的`padding`参数介绍过）此时的掩码可能形如：

```text
tensor([[1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1]], device='cuda:0')
```

**此时应该传递掩码进去**

`do_sample`：默认为**True**，为**True**时正常采样，为**False**时每一步都会选取概率最高的**token**

`num_return_sequences`：默认为**1**，此参数用于指定每个样本的续写数量，举例来说，其为一的时候输出的**generated_ids**的**shape**若为`torch.Size([2, 58])`，则其为三的时候为`torch.Size([6, 58])`，其中沿着**batch_size**维度，前三个对应第一个序列的生成结果

***



