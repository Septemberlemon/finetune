## **datasets库是hugging face出品的专门用于数据加载、处理的库**

使用`pip install datasets`即可安装

***

### 加载在线数据集

使用`load_dataset`接口进行数据集的加载，如：

```python
from datasets impot load_dataset

dataset = load_dataset("repo_id")
```

其中`repo_id`是**hugging face**的数据集仓库名，可以直接从官网拷贝

上述代码将自动从官网下载数据集到本地`~/.cache/huggingface/hub`下

***

### 关于`load_dataset`得到的对象

其一般是一个专门定义的`DatasetDict`或`Dataset`对象，具体是哪个取决于在线数据集是否进行了**预定义的分片**（这取决于官网中的仓库内容），例如使用`load_dataset("ylecun/mnist")`得到的对象是一个`DatasetDict`：

```shell
DatasetDict({
    train: Dataset({
        features: ['image', 'label'],
        num_rows: 60000
    })
    test: Dataset({
        features: ['image', 'label'],
        num_rows: 10000
    })
})
```

其中包含了**"train"**和**"test"**两个`Dataset`对象，可以像普通对象一样通过键名取出一个`Dataset`对象：

```python
mnist_dataset["train"]
```

```shell
Dataset({
    features: ['image', 'label'],
    num_rows: 60000
})
```

其中每一个`Dataset`对象都可以视作一个**“表格”**，其中：

* **每一行** 代表一个 **独立样本**
* **每一列** 代表一种 **特征（feature）**

通过`Dataset.num_rows`查看行数，`Dataset.features`查看所有特征名

可以通过**数字索引**获取行，通过**特征名**索引获取列；前者将拿到一个**python dict**类型的变量作为一个独立样本，后者一般会拿到一个专门定义的`Column`对象，如：

```python
train_mnist_dataset[0]
```

```shell
{'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28>,
 'label': 5}
```

```python
train_mnist_dataset["images"]
```

```shell
Column([<PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x7F198C2C4AF0>, <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x7F198C2C4C70>, <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x7F198C2C56C0>, <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x7F198C2C57E0>, <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x7F198C2C57B0>])
```

接下来可以使用两种不同的方式取到某个样本内的一个**feature**：

```python
train_mnist_dataset[1000]["image"]
```

```python
train_mnist_dataset["image"][1000]
```

它们都能拿到内部的`PIL.PngImagePlugin.PngImageFile`对象

***

### 从本地文件加载数据集

使用`load_dataset`加载本地数据集，第一个参数根据本地数据集所在的**文件格式**确定，例如对于**json**或**jsonl**文件存储的数据集，可以使用下述代码加载数据集：

```python
local_dataset = load_dataset("json", data_files="/home/u/finetune/data/bad_woman/train.json")
```

**注意data_files参数一定要作为关键字参数给出，其值为本地文件路径**

上述代码将得到下述对象：

```shell
DatasetDict({
    train: Dataset({
        features: ['conversations'],
        num_rows: 537
    })
})
```

可以通过取其**"train"**键得到数据集本身

之所以得到的是`DatasetDict`对象而非一个直接的`Dataset`对象，见下述对`split`参数的说明

***

### split参数

这是`load_dataset`接口的一个**关键字参数**，其默认值为`None`，用于指定从**数据集（`DatasetDict`）**中取出特定的一个**分片（`Dataset`）**，例如说**mnist**数据集中包含**”train"**和**“test”**两个分片，则可以使用代码：

```python
test_mnist_dataset = load_dataset("ylecun/mnist", split="test")
```

直接拿到**mnist**数据集中的**test**部分：

```shell
Dataset({
    features: ['image', 'label'],
    num_rows: 10000
})
```

这意味着得到的`test_mnist_dataset`将是一个直接的`Dataset`对象，无需像`DatasetDict`那样通过键名取到`Dataset`对象

对于从**一个本地文件**中加载的数据集来说，它会将这整个文件中的数据加载为一个**单独的分片（`Dataset`）**，并将其命名为**"train"**放进一个`DatasetDict`中，这解释了为什么前面加载本地数据集的代码拿到的是一个`DatasetDict`对象，而根据`split`参数的作用，只需要在前面的代码中加上`split="train"`即可直接拿到`Dataset`：

```python
local_dataset = load_dataset("json", data_files="/home/u/finetune/data/bad_woman/train.json", split="train")
```

```shell
Dataset({
    features: ['conversations'],
    num_rows: 537
})
```

***

### data_files参数

此参数用于指定数据集文件路径，最简单的用法见上述**加载本地数据集**的部分，指定其为文件路径即可

也可以指定其为多个文件，如：

```python
files = {
    "train": "/home/u/finetune/data/bad_woman/train.json",
    "eval": "/home/u/finetune/data/bad_woman/eval.json"
}
local_dataset = load_dataset("json", data_files=files)
```

其将得到：

```shell
DatasetDict({
    train: Dataset({
        features: ['conversations'],
        num_rows: 537
    })
    eval: Dataset({
        features: ['conversations'],
        num_rows: 80
    })
})
```

注意，如果在上述代码中指定了`split="train"`：

```python
files = {
    "train": "/home/u/finetune/data/bad_woman/train.json",
    "eval": "/home/u/finetune/data/bad_woman/eval.json"
}
local_dataset = load_dataset("json", data_files=files, split="train")
```

根据`split`参数的作用，将会从`DatasetDict`中取出**"train"分片**，最终得到的还是：

```shell
Dataset({
    features: ['conversations'],
    num_rows: 537
})
```

也可以为每个**分片**指定多个对应的文件，**datasets**将自动将把被指定的文件们中的数据合并到一个`Dataset`中，如：

```python
files = {
    "train": [
        "/home/u/finetune/data/bad_woman/train1.json",
        "/home/u/finetune/data/bad_woman/train2.json",
    ],
    "eval": "/home/u/finetune/data/bad_woman/eval.json"
}
local_dataset = load_dataset("json", data_files=files, split="train")
```

或者采用通配符（注意下述代码中的星号），这将加载特定目录下所有匹配的文件并合并：

```python
files = {
    "train": "/home/u/finetune/data/bad_woman/train*.json",
    "eval": "/home/u/finetune/data/bad_woman/eval.json"
}
local_dataset = load_dataset("json", data_files=files, split="train")
```

此外，`data_files`参数也支持 **url**，如：

```python
from datasets import load_dataset

# 混合使用本地路径和远程 URL
files_map_mixed = {
    "train": "https://huggingface.co/datasets/lhoestq/demo1/raw/main/data.csv", # 远程文件
    "test": "/path/to/local/test_data.csv"                                     # 本地文件
}

dataset_dict = load_dataset("csv", data_files=files_map_mixed)
```

***

### 从一个数据集中划分不同分片

有时候需要手动将一个数据集进行划分，假设我们已经有了一个`Dataset`对象**dataset**，则可以调用`train_test_split`方法进行划分，如：

```python
split_datasets = dataset.train_test_split(test_size=0.2, seed=37)
```

这将得到一个划分后的`DatasetDict`，其中包含名为**"train"**和**"test"**的两个`Dataset`：

```shell
DatasetDict({
    train: Dataset({
        features: ['image', 'label'],
        num_rows: 8000
    })
    test: Dataset({
        features: ['image', 'label'],
        num_rows: 2000
    })
})
```

可以使用`pop`方法移除`DatasetDict`里的`Dataset`，这同时会将其返回，例如对于上述得到的`split_datasets`使用：

```python
test_dataset = split_dataset.pop("test")
```

将会得到：

```python
split_datasets
```

```shell
DatasetDict({
    train: Dataset({
        features: ['image', 'label'],
        num_rows: 8000
    })
})
```

```python
test_dataset
```

```shell
Dataset({
    features: ['image', 'label'],
    num_rows: 2000
})
```

使用`pop`可对`DatasetDict`中的`Dataset`进行重命名，例如对于上述的`split_datasets`，使用：

```python
split_datasets["eval"] = split_datasets.pop("train")
```

将会得到：

```python
split_datasets
```

```shell
DatasetDict({
    eval: Dataset({
        features: ['image', 'label'],
        num_rows: 8000
    })
})
```

***

### 图像分类数据集的加载

我们预期一个良好的文件夹结构，如：

```text
my_image_dataset/
├── train/
│   ├── cat/
│   │   ├── 001.jpg
│   │   ├── 002.png
│   │   └── ...
│   └── dog/
│       ├── 101.jpg
│       ├── 102.jpeg
│       └── ...
└── test/
    ├── cat/
    │   ├── 201.jpg
    │   └── ...
    └── dog/
        ├── 301.jpg
        └── ...
```

使用下述代码进行加载：

```python
from datasets import load_dataset

# data_dir 指向包含 train/ 和 test/ 的那个根目录
image_dataset = load_dataset("imagefolder", data_dir="/path/to/my_image_dataset")

print(image_dataset)
```

这将得到：

```shell
DatasetDict({
    train: Dataset({
        features: ['image', 'label'],
        num_rows: ... # train 文件夹下所有图片的数量
    })
    test: Dataset({
        features: ['image', 'label'],
        num_rows: ... # test 文件夹下所有图片的数量
    })
})
```

其中**"label"**标签为**"cat"**、**"dog"**这些子文件夹名对应的一个整数，具体的映射关系取决于字母顺序，可以通过：

```python
labels = image_dataset["train"].features["label"].names
```

来查看排序，如：

```shell
['bird', 'cats', 'dogs', 'fox']
```

通过：

```python
example = image_dataset['train'][0]
print(example['label'])  # 输出 0
print(labels[example['label']])  # 输出 'birds'
```

可查看特定样本原本所属的子文件名，也就是类别

或者你有一个没有划分**分片**的文件夹结构：

```text
my_images_no_splits/
├── cat/
│ ├── 001.jpg
│ ├── 002.png
│ └── ...
├── dog/
│ ├── 101.jpg
│ ├── 102.jpeg
│ └── ...
└── bird/
├── 201.jpg
└── ...
```

则可以使用下述代码进行加载：

```python
from datasets import load_dataset

# 加载整个文件夹
full_image_dataset_dict = load_dataset("imagefolder", data_dir="/path/to/my_images_no_splits")

print("--- 初始加载的对象 ---")
print(full_image_dataset_dict)
```

这将得到一个包含一个名为**"train"**的`Dataset`的`DatasetDict`：

```shell
DatasetDict({
    train: Dataset({
        features: ['image', 'label'],
        num_rows: ... # 所有图片的总数
    })
})
```

后续可以通过前面介绍过的方法手动划分**分片**

***

### map方法

在加载了原始数据集之后往往需要对其进行一些处理以便使用，这往往使用`map`方法。当然也可以用**pandas**处理，然后使用：

```python
from datasets import Dataset

data = [[1, 2, 3], [4, 5, 6]]
dataset = Dataset.from_pandas(pd.DataFrame(data))
```

从一个`DataFrame`中加载`Dataset`

`map`方法由`Dataset`对象调用，接收一个函数作为参数，同时往往指定一个关键字参数**batched**（默认为False），例如：

```python
dataset = dataset.map(formatting_prompts_func, batched=True)
```

上述代码将会把`formatting_prompts_func`这个函数作用于`Dataset`对象上，返回一个作用后的新`Dataset`对象

不妨称被**map**作用的函数为**format**，**format**函数接收一个参数，**map**方法执行的时候，会多次调用**format**函数，向其内部传参。传参的内容会因为**batched**参数的不同而不同，具体来说，当**batched**为**False**时，**format**函数接收的参数是一个`datasets.formatting.formatting.LazyRow`对象（类似字典），它包含的是数据集中单个样本的信息；当**batched**为**True**时，**format**函数接收的参数是一个`datasets.formatting.formatting.LazyBatch`对象，它包含的是数据集中一批样本的信息，所以可以根据**batched**参数的不同，采用下述两种形式的函数：

```python
def format(examples):
	pass
```

```python
def format_single(example):
    pass
```

`examples`参数名暗示了它是一个**"batch"**，而`example`参数名则暗示了它是**单个样本**

然后分别采用下述方式调用：

```python
formatted_dataset = dataset.map(format, batched=True)
```

```python
forrmatted_dataset = dataset.map(format_single, batched=False)
```

下面分别介绍**batched=True**和**batched=False**时的被**map**的函数内部应该怎么处理

#### batched=False

内部的参数`example`类型为`datasets.formatting.formatting.LazyRow`，代表一个独立的样本，可以当作字典使用，通过**特征名**可以索引样本中特定特征的值，并用其进行运算。函数最后应该返回一个字典或者`datasets.formatting.formatting.LazyRow`。在函数内部对样本进行的修改将会生效，函数返回的字典或`datasets.formatting.formatting.LazyRow`将会被被合并到到样本中（**若有重复键则覆盖**），如：

```python
def format_single(example):
    return {"text_length": len(example["conversations"])}
```

或者：

```python
def format_single(example):
    example['text_length'] = len(example['conversations'])
    return example
```

都可以为样本添加`text_length`字段

**注意若是函数返回值为*None*或者无返回值，函数内部对样本的修改将无效**

**建议使用返回新字段的方式进行修改而非在函数内部修改，以保持清晰的结构**

#### batched=True

内部的参数`examples`类型为`datasets.formatting.formatting.LazyBatch`，代表一批样本，也可以当作字典使用。不同的是使用**特征名**索引将得到一个列表对象，列表中每个元素为特定特征的值。函数最后应该返回一个字典或者`datasets.formatting.formatting.LazyBatch`，如果返回字典则应该保持字典的值是长度为`len(examples)`的列表，字典键将作为数据集中新特征的键名。在函数内部对**样本批**进行的修改也将生效，函数返回的字典或`datasets.formatting.formatting.LazyBatch`将会被合并到样本中（**若有重复键则覆盖**），如：

```python
def format(examples):
    conversations = examples["conversations"]
    text_length = [len(conversation) for conversation in conversations]
    return {"text_length": text_length}
```

或者：

```python
def format(examples):
    conversations = examples["conversations"]
    examples["text_length"] = [len(conversation) for conversation in conversations]
    return examples
```

都能为样本添加`text_length`字段

**注意若是函数返回值为*None*或者无返回值，函数内部对样本批的修改将无效**

**仍然建议使用返回新字段的方式以保持代码结构清晰**

若要移除一些字段，可以使用**map**函数的`remove_columns`参数，用`str`或者`list[str]`指定要删除的键名即可，**该参数只能用于指定数据集中原本拥有的字段**

`num_proc`是**map**函数的一个用于指定应用**map**函数时的处理进程数的参数，增大它有助于加速数据处理

**建议一直指定`batched=True`，因为它更快：函数调用次数少导致开销小、有一些专门用于快速批量处理的接口可供调用以提速**

### set_format和set_transform方法，以及DataLoader

使用`map`方法会将**map**后的数据集存在磁盘上导致占用激增，可以使用动态转换的方法：`set_format`和`set_transform`，它们将会在实际遍历数据集时对样本进行处理，牺牲时间来换空间

`set_format`方法由`Dataset`对象调用，是**原位方法**，例如：

```python
mnist_dataset.set_format(type="torch", columns=["image", "label"])
```

会将`image`、`label`的类型从分别从`PIL.PngImagePlugin.PngImageFile`、`int`转为`torch.tensor`，

转换的列由`columns`参数指定，它要求一个由列名构成的**list**

转换的类型由`type`参数指定，常见的可选值有**torch**、**numpy**、**tensorflow**、**pandas**

**`set_format`能转换的类型有限，而且不能做其他处理**

`set_transform`方法也是**原位方法**，由`Dataset`对象调用，它接受一个可调用对象作为参数，例如：

```python
import torchvision.transforms as T

augmentations = T.Compose([
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(), # 将 PIL Image 转换为 PyTorch Tensor (0-1 a 范围内)
    T.Normalize(mean=[0.5], std=[0.5]) # 归一化到 -1 到 1
])

def transform(batch_dict):
    batch_dict["image"] = [augmentations(image) for image in batch_dict["image"]]
    return batch_dict

train_mnist_dataset.set_transform(transform)
```

上述代码对原始图像进行了一些随机的变换，并归一化并转为**tensor**

对于被`set_transform`回调的函数，其内部接收的参数为一个**python dict**，其中的键为数据集中的**所有字段名**（也可以通过指定`set_transform`的`columns`参数进行限定，该参数为要参与变换的字段名字符串构成的列表），值为对应字段的若干元素构成的**list**（该**list**长度取决于遍历的方法，例如普通的**for**循环将会传长度为**1**的列表进去，用`DataLoader`（`Dataset`对象本身是兼容**torch**的`DataLoader`的，因为它内部实现了`__getitem__`方法和`__len__`方法）将会传递长度为指定`batch_size`的列表进去），返回值要求一个类似于输入参数的**python dict**，它将决定遍历数据集的时拿到的结果。

对于上述例子，它返回的结果和传进参数结构基本一致，只是把`"image"`字段列表中的每一个`PIL.PngImagePlugin.PngImageFile`对象做了一些图像变换并转换为了`tensor`，外部遍历此数据集的时候每次拿到的将会取决于函数内部直接`return`的值，具体来说：

* 对于普通的`for`循环，每次拿到的**sample**将会是一个类似函数内`return`值的字典，只是把作为值的两个长度为**1**的列表中的唯一一个值取了出来，相当于去掉了外层的列表；
* 对于`DataLoader`遍历数据集的时候，每次拿到的**batch_samples**也是一个类似于函数内`return`值的字典，但其内部会对字典的值进行堆叠并转换为`tensor`，这意味着`"image"`的值将会被**stack**为一个**shape**为**[batch_size, 1, 28, 28]**的**tensor**（回调函数内部返回的是**batch_size**个**shape**为**[1, 28, 28]**的**tensor**构成的列表），`"label"`的值将会被转换为一个**shape**为**[batch_size]**的**tensor**（回调函数内部返回的是**batch_size**个`int`构成的列表）

若要移除这两种动态转换的方法，使用`dataset.reset_format()`接口

***

### 数据集的保存和再加载

可以将处理过的或是从远程下载的数据集保存到本地特定文件夹，再次使用时直接加载：

```python
# 将处理后的数据集保存到磁盘
processed_dataset.save_to_disk("/path/to/my_processed_dataset")

# 将下载来的数据集保存起来
mnist_dataset.save_to_disk("/path/to/somewhere")
mnist_dataset["train"].save_to_disk("/path/to/somewhere_else")
```

注意，`save_to_disk`保存起来的对象可以是`Dataset`或者`DatasetDict`，它们用同名的`save_to_disk`接口，但是**加载的时候要用不同的接口**：

```python
from datasets import Dataset, DatasetDict

# 加载 DatasetDict 和 Dataset 要用不同的接口
mnist_dataset = DatasetDict.load_from_disk("/path/to/somewhere")
train_mnist_dataset = Dataset.load_from_disk("/path/to/somewhere_else")
```

因为有些下载来的数据虽然已经缓存在了本地，但是因为缺少配置文件等原因，无法直接使用`load_from_disk`加载，所以可以另外保存，就能离线加载了

也可以将`Dataset`对象保存为本地**csv**文件或者**json**文件，使用：

```python
# 转换为 CSV 或 JSON 文件
dataset.to_csv("/path/to/output.csv")
dataset.to_json("/path/to/output.json")
```

***

### 从python对象中加载数据集

这通常用于演示或者快速验证：

```python
from datasets import Dataset

# 创建一个字典，其中每个键对应一列数据
data = {
    "text": ["这是一个句子。", "这是另一个句子。", "最后一个例子。"],
    "label": [1, 0, 1]
}

# 从字典创建数据集
dataset = Dataset.from_dict(data)

# 打印数据集信息
print(dataset)
# 输出:
# Dataset({
#     features: ['text', 'label'],
#     num_rows: 3
# })

# 查看第一行数据
print(dataset[0])
# 输出:
# {'text': '这是一个句子。', 'label': 1}
```

```python
from datasets import Dataset

# 创建一个字典的列表
data = [
    {"text": "这是一个句子。", "label": 1},
    {"text": "这是另一个句子。", "label": 0},
    {"text": "最后一个例子。", "label": 1}
]

# 从列表创建数据集
dataset = Dataset.from_list(data)

# 打印数据集信息
print(dataset)
# 输出:
# Dataset({
#     features: ['text', 'label'],
#     num_rows: 3
# })

# 查看第一行数据
print(dataset[0])
# 输出:
# {'text': '这是一个句子。', 'label': 1}
```