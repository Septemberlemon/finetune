## 安装

使用以下项目进行**Linux**端代理配置：[https://github.com/nelvko/clash-for-linux-install.git](https://github.com/nelvko/clash-for-linux-install.git)

首先依次执行以下两条命令，**clone**项目并进入目录：

```shell
git clone https://github.com/nelvko/clash-for-linux-install.git
```

```shell
cd clash-for-linux-install
```

下一步可以使用**订阅文件**或者**订阅url**进行配置，

#### 首先是采用**订阅url**的方式：

直接执行以下命令：

```shell
sudo bash install.sh
```

根据提示完成操作（包括**订阅url**的输入）即可。
#### 然后是采用**订阅文件**的方式

将订阅文件手动拷贝到项目目录下的`resources`目录中，并重命名为`config.yaml`，再回到项目**根目录**，执行以下命令即可：

```shell
sudo bash install.sh
```

***

## 常用指令

`clashctl`：查看相关命令

`clashon` `clashctl on`：打开代理

`clashoff` `clashctl off`：关闭代理

`clashupdate` `clashctl update`：更新代理配置文件

`clashstatus` `clashctl status`：查看当前代理状态

`clashui` `clashctl ui`：查看**web ui**相关信息

`clashsecret` `clashctl secret`：查看**web**控制台密钥

***

## Web控制台

首先执行`clashui`命令查看输出，例如：

```shell
╔═══════════════════════════════════════════════╗
║                😼 Web 控制台                   ║
║═══════════════════════════════════════════════║
║                                               ║
║     🔓 注意放行端口：9090                        ║
║     🏠 内网：http://172.23.96.49:9090/ui       ║
║     🌏 公网：http://223.73.115.56:9090/ui      ║
║     ☁️  公共：http://board.zash.run.place      ║
║                                               ║
╚═══════════════════════════════════════════════╝
```

访问其中的**内网url**（此例中为`http://172.23.96.49:9090/ui`），到达**Web控制台**的登录界面；

执行`clashsecret`命令查看登录密钥，复制粘贴到**Secret**栏；

再将上述`clashui`命令输出中的**内网url**中的**base**部分（此例中为`http://172.23.96.49:9090`）粘贴到**API Base URL**栏，点击**Add**，再点击下方新增的按钮上的**url**即可登录**Web控制台**。

退出登录位置在左侧**Config**选项卡的**Switch Backend**按钮。