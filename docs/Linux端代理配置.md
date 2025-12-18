## 安装

使用以下项目进行**Linux**端代理配置：[https://github.com/nelvko/clash-for-linux-install.git](https://github.com/nelvko/clash-for-linux-install.git)

首先依次执行以下两条命令，**clone**项目并进入目录：

```shell
git clone https://github.com/nelvko/clash-for-linux-install.git
```

```shell
cd clash-for-linux-install
```

接下来编辑**.env**文件，指定内核版本和安装位置，前者建议使用新版的**mihomo**，后者指定为当前用户拥有权限的目录即可，如当前用户家目录下的某个路径，不要在路径中使用`~`，否则下一步的安装会使用`sudo`提权可能会导致错误

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
***

## 一些额外说明

如果安装的是**mihomo**内核，那么实际上底层启动的是**mihomo.service**，可以查看它的状态

使用`clashon`实际上做了两件事，一是检查这个服务是否开启，若未开启则开启之、二是将代理相关的环境变量加载到**shell**中，`clashoff`则是对应的关闭服务和擦除环境变量

这意味着在一次登陆后执行`clashon`之后，若未执行`clashoff`即断开连接，**mihomo.service**实际上仍然在运行，只是环境变量随**shell**一同消散了，下次登陆后，**mihomo.service**仍然在运行，只是**shell**因为是新加载的，所以里面没有代理相关的环境变量，此时执行`clashon`即可写入环境变量

对于长期运行的**service**，可以在代码内手动指定**proxy**为**mihomo**对应的代理端口，或者在**service**配置文件中写明

对于继承自**shell**的程序，例如在**shell**中执行**python**，其代理相关环境变量将也被继承过去，这点需要注意，另外若将**python**配置成**service**则不会继承**shell**的环境变量

`clashon`本身是一个**shell**函数，它被`clashctl.sh`加载，在安装完成后可以查看**.bashrc**文件，底部将有一个加载`clashctl.sh`的部分，它实际上就是把其内的各个函数加载到**shell**中。每次一个新**shell**启动之后，**.bashrc**文件将被加载，`clashctl.sh`随之加载，`clashon`等函数随之被写入