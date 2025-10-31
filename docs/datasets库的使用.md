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

可以通过**数字索引**获取行，通过**特征名**索引获取列；前者将拿到一个**python dict**类型的变量作为一个独立样本，后者将拿到一个专门定义的`Column`对象，如：

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

### 加载本地数据集

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

上述代码将会把`formatting_prompts_func`这个函数作用于`Dataset`对象上，返回一个作用后的新对象

不妨称被**map**作用的函数为**format**，**format**函数接收一个参数，**map**方法执行的时候，会多次调用**format**函数，向其内部传参。传参的内容会因为**batched**参数的不同而不同，具体来说，当**batched**为**False**时，**format**函数接收的参数是一个`datasets.formatting.formatting.LazyRow`对象，它包含的是数据集中单个样本的信息；当**batched**为**True**时，**format**函数接收的参数是一个`datasets.formatting.formatting.LazyBatch`对象，它包含的是数据集中一批样本的信息，所以可以根据**batched**参数的不同，采用下述两种形式的函数：

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
dataset.map(format, batched=True)
```

```python
dataset.map(format_single, batched=False)
```

下面分别介绍**batched=True**和**batched=False**时的被**map**的函数内部应该怎么处理

#### batched=False

内部可以直接将`example`作为一个独立的样本使用，通过**特征名**索引样本中特定特征的值，并用其进行运算，最后返回一个字典。在函数内部对样本进行的修改将会生效，后续返回的字典将会被被合并到

#### batched=True