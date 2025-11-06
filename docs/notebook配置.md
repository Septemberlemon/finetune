## 安装jupyter notebook

激活虚拟环境后直接执行`pip install notebook`即可

## 设置访问密码

在虚拟环境激活情况下执行`jupyter notebook password`，按照系统提示设置密码即可，此密码会被用来在网页端登录时使用

## 生成配置文件并编辑

在虚拟环境激活情况下执行`jupyter notebook --generate-config`，这会在路径`~/.jupyter`下生成一个名为`jupyter_notebook_config.py`的文件。

编辑该文件并在末尾添加如下四行

```
c.NotebookApp.ip = '0.0.0.0'  # 允许任何 IP 地址访问
c.NotebookApp.port = 8899  # 指定一个未被占用的端口号
c.NotebookApp.open_browser = False  # 禁止在服务器上自动打开浏览器
c.NotebookApp.allow_remote_access = True # 允许远程连接
```

执行完此步后即可使用命令`jupyter notebook`启动**notebook**。
启动之后，可通过在浏览器中输入**url**`http://192.168.10.166:8899`来访问**notebook**，**url**中的地址为安装**notebook**的**主机ip**。

## 编辑系统服务文件

在路径`/etc/systemd/system`下创建文件`notebook.service`并编辑，输入以下内容：

```
[Unit]
Description=jupyter notebook service
After=network.target

[Service]
User=u
WorkingDirectory=/home/u/finetune
ExecStart=/home/u/finetune/.venv/bin/jupyter-notebook

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

保存后，执行`systemctl daemon-reload`之后即可启动服务

## 代理设置

确保主机上代理为开启状态。

执行以下两条命令查看系统代理：
```shell
echo $HTTP_PROXY
echo $HTTPS_PROXY
```

输出的两个字符串将被用来配置**notebook**的代理，在`notebook.service`的`[Service]`字段添加如下两行：

```
Environment="HTTP_PROXY=http://127.0.0.1:7890"
Environment="HTTPS_PROXY=http://127.0.0.1:7890"
```

第一行中的`http://127.0.0.1:7890`为`echo $HTTP_PROXY`的输出
第二行中的`http://127.0.0.1:7890`为`echo $HTTPS_PROXY`的输出
这里它们值是一样的。

## Unsloth离线运行相关设置

为了让**unsloth**可以离线运行（有时这是必要的，因为代理有时候不能正常工作），需要添加一些环境变量。

在`notebook.service`的`[Service]`字段添加如下三行：

```
Environment="UNSLOTH_DISABLE_STATISTICS="""
Environment="HF_HUB_OFFLINE="1""
Environment="TRANSFORMERS_OFFLINE="1""
```

相关网址：

[https://huggingface.co/docs/transformers/main/installation#offline-mode](https://huggingface.co/docs/transformers/main/installation#offline-mode)

[https://github.com/unslothai/unsloth/blob/69a64758e56fe94f103cb00da078db27ff886cf3/unsloth/models/_utils.py#L1001](https://github.com/unslothai/unsloth/blob/69a64758e56fe94f103cb00da078db27ff886cf3/unsloth/models/_utils.py#L1001)

需要注意的是如果设置了上述环境变量会在下载新模型时导致问题，建议使用代码下载新模型时手动先在代码中删除这两个环境变量，或者使用**huggingface cli**下载模型。

## 一些常用快捷键

首先**notebook**有两种模式：**编辑模式**、**命令模式**
前者就是输入一个字符就在一个**cell**中出现一个字符的模式，后者则是悬浮在某一个**cell**而没有聚焦到其内部开始编辑的模式。
`Enter`：当处于**命令模式**时，悬浮在某个**cell**上时，点击`Enter`转换为**编辑模式**，并将光标置于其内
`Esc`：当处于**编辑模式**时，点击`Esc`转换为**命令模式**，悬浮在此**cell**上
`Shift+j`：当处于**命令模式**时，点击`Shift+j`将悬浮位置移动到当前**cell**的**下一个*cell***
`Shift+k`：当处于**命令模式**时，点击`Shift+k`将悬浮位置移动到当前**cell**的**上一个*cell***
`a`：当处于**命令模式**时，点击`a`在当前**cell**的上方**插入**一个新**cell**，并将悬浮位置移动到其上
`b`：当处于**命令模式**时，点击`b`在当前**cell**的下方**插入**一个新**cell**，并将悬浮位置移动到其上
`dd`：当处于**命令模式**时，**双击**`d`删除当前**cell**，并将悬浮位置移动到下一个**cell**
`ii`：当处于**命令模式**且当前**cell**正在运行时，**双击**`i`中止当前**cell**的运行
`Shift+Enter`：当处于**命令模式**或者**编辑模式**时，点击`Shift+Enter`运行当前**cell**的代码
`Home`：当处于**编辑模式**时，点击`Home`将光标移至行首
`End`：当处于**编辑模式**时，点击`End`将光标移至行尾