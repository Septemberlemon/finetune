### 引言

**huggingface-cli** 是*Hugging Face*官方提供的命令行工具，通过它可以直接使用命令行与*Hugging Face Hub*交互。

首先要安装`huggingface_hub`库，使用命令：
```shell
pip install huggingface_hub
```

安装，之后即可使用**huggingface-cli**命令。
**新版本的`huggingface-cli`命令已经过时了，它仍然可以使用，但是建议使用`hf`命令代替**

### 登录

使用命令：

```shell
hf auth whoami
```

查看当前登录用户，若未登录，可使用命令：

```shell
hf auth login
```

进行登录，运行命令后出现登录提示，输入**token**即可登录
**token**需要去[Hugging Face 官网](https://huggingface.co/)登录后获取，具体位置为：*右上角头像* -> *Settings* -> *左侧 Access Tokens*
登录过后的**token**将被保存在`~/.cache/huggingface/token`文件中

### 下载模型

使用命令：
```shell
hf download repo_id
```

下载模型，例如：

```shell
hf downlaod unsloth/Qwen3-14B
```

`repo_id`即为模型仓库名，可直接从官网拷贝。
下载来的模型的**保存位置**为`~/.cache/huggingface/hub`