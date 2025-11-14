# 安装

由于版本冲突的原因，建议在全新的环境安装**vllm**

先使用命令：

```shell
pip install uv
```

安装**uv**，再使用命令：

```shell
uv pip install vllm --torchbackend=auto
```

进行安装。此命令将自行检测系统信息，安装符合版本要求的**torch**依赖

如果要指定特定版本的**torch**依赖，例如**cuda 12.6**版本的torch，使用命令：

```shell
uv pip install vllm --torchbackend=cu126
```

# vllm cli的使用

## 基础

运行`vllm` `vllm -h` `vllm --help`来查看基础的**vllm**命令行相关参数，可以得知它有**6**个子命令：

* **chat**
* **complete**
* **serve**
* **bench**
* **collect-env**
* **run-batch**

运行`vllm -v` `vllm --version`可以查看**vllm**版本。

## 关于帮助

使用`vllm [subcommand] --help=all`查看某一子命令**全部选项说明**，例如：

```shell
vllm serve --help=all
```

可以查看**serve**子命令的全部选项说明

***

使用`vllm [subcommand] --help`查看某一子命令的**全部选项组**，每个选项组包含若干选项。

***

使用`vllm [subcommand] --help=[option-group]`查看某一子命令的**某一选项组的选项说明**，例如：

```shell
vllm serve --help=loraconfig
```

查看**serve**子命令的**LoRAConfig**选项组中的若干选项说明，*选项组名称大小写不敏感*。

***

使用`vllm [subcommand] --help=[option]`查看某一个子命令的**匹配选项的说明**，其会列出与**option**字符串相匹配的所有选项，例如：

```shell
vllm serve --help=enable-lora
```

会得到下述输出：

```shell
Arguments matching 'enable-lora':
  --enable-lora, --no-enable-lora
                        If True, enable handling of LoRA adapters. (default: None)
  --enable-lora-bias, --no-enable-lora-bias
                        [DEPRECATED] Enable bias for LoRA adapters. This option will be removed in v0.12.0. (default: False)
```

*选项名称大小写不敏感，且可以使用`_`替换`-`*。

[**vllm cli**的官方文档](https://docs.vllm.ai/en/latest/cli/index.html)

## vllm serve

使用`vllm serve`命令来启动**vllm**服务后端，基础参数`model_name`，例如：

```shell
vllm serve unsloth/Qwen3-32B
```

即会从`huggingface hub`下载对应模型到本地并加载，下载位置与**transformers**库下载位置一样：`~/.cache/huggingface/hub`

若要加载本地模型，例如`unsloth/Qwen3-32B`，则需要将模型文件夹中的**snapshots/[commit_hash]**路径作为`model_name`参数值：

```shell
vllm serve /home/u/.cache/huggingface/hub/models--unsloth--Qwen3-32B/snapshots/544fef601b113237a39f3493a579846a32bdc414
```

上述命令中的`544fef601b113237a39f3493a579846a32bdc414`即为提交**hash**值，**snapshots**文件夹下只有其一个子文件夹。

#### 模型对外名称指定

使用选项`--served-model-name`选项指定**此模型对外部api提供的名称**，若不提供，则默认与`model_name`相同。

指定该选项之后，`model_name`将不再作为模型对外部**api**提供的名称。

对于本地加载的模型，若不指定此值，会导致模型对外部**api**提供的名称含有本地路径，所以往往指定一个更简短的名称，例如：

```shell
vllm serve /home/u/.cache/huggingface/hub/models--unsloth--Qwen3-32B/snapshots/544fef601b113237a39f3493a579846a32bdc414 --served-model-name Qwen3-32B
```

#### 加载LoRA模型

使用选项`--enable-lora`启动**LoRA**，并使用选项`--lora-modules`指定**LoRA**模型名称及位置，如：

```shell
vllm serve /home/u/.cache/huggingface/hub/models--unsloth--Qwen3-32B/snapshots/544fef601b113237a39f3493a579846a32bdc414 --served-model-name Qwen3-32B --enable-lora --lora-modules bad-woman=/home/u/finetune/lora_model/bad_woman
```

上述例子将**LoRA**模型的名称指定为**bad-woman**，并指定了本地的**LoRA**模型所在文件夹。

**注意，运行上述命令后，将对外部提供两个模型api，Qwen3-32B 和 bad-woman，前者是不加载 LoRA 模型的接口，后者是加载 LoRA 模型的接口。**

**若指定二者相同，则加载 LoRA 模型的接口优先。**

#### 指定聊天模板

**vllm**默认使用模型所处文件夹的聊天模板，并会在内部进行缓存，所以需要手动进行更改以实现自定义聊天模板，使用选项`--chat-template`，指定其值为聊天模板的文件路径，即可使用自定义聊天模板对输入进行处理