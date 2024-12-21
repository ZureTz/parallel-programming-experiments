> 本文为中国石油大学（北京）并行程序设计课程大作业而准备，内容包括从源码安装PyTorch的指南，以及代码模板的说明

# 从源码安装PyTorch

源码安装的官方说明 [PyTorch Github | From Source](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source) ； 
中文社区较完善的参考 [CSDN | PyTorch编译流程踩坑](https://blog.csdn.net/qq_36755743/article/details/136877045)

## 1 - 环境准备 安装Anaconda

```shell
apt update  # 更新ubuntu的源  
apt install git vim wget # 安装 必要的一些工具
```

其他环境版本等应符合要求

```shell
gcc -v
g++ -v  # 分别检查版本信息
nvcc -V # cuda 安装的版本情况
nvidia-smi # 驱动情况，以及可以支持的最高cuda
```
先检查 `which python3.10` 是否能够检测到python，如果能检测到则跳过python安装步骤

该镜像没有python应手动安装一个，选择python3.10因为在之前拉取的成熟镜像中，PyTorch2.5.0 + cuda12.6都可以适配python3.10

```shell
apt install python3.10 # 在系统依赖中安装python 
apt install pip  # 安装pip工具 --- 安装好了可以直接 which pip 检测到
```

此时使用 `which python`未能检测到python，但使用 `which python3.10`可以；因为没有创建链接为`python`

```shell
which python3.10 # 输出结果为 /usr/bin/python3.10
ln -s /usr/bin/python3.10 /usr/bin/python # 创建 python 软链接
python --version # 安装完成检查版本
```

安装anaconda，并[换第三方conda源到清华源](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)；参考教程 [安装Anaconda详细步骤](https://blog.csdn.net/thy0000/article/details/122878599#:~:text=%E5%9C%A8Ubuntu%E4%B8%8A%E5%AE%89%E8%A3%85)

```shell
wget --user-agent="mozilla" https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.06-1-Linux-x86_64.sh   # 下载清华源上的anaconda，清华源是在北方地区较完善、较快的源；其他源推荐：中科大源
```

按照提示安装 Anaconda，基本上一路enter + yes就可以；

如果安装完了以后在bash中conda无法识别到，即`conda list`输出`bash: conda: command not found` ；首先应尝试重新执行上面`xxx.sh`脚本，查看安装时输出信息，检查`conda list`输出情况。还是不行，则应在环境变量中声明：

```shell
echo 'export PATH="/home/parallel_X//bin:$PATH"' >> ~/.bashrc  # 添加到环境变量（注意：替换为自己的路径）
source ~/.bashrc  # 更新环境变量
```

## 2 - 下载文件 创建环境

参考官方信息，git下来代码源文件，文件较多，安装时间较长
（fatal的话可能是网络不好，可以多试几次，实在不行可能需要梯子）

```shell
git clone --recursive https://github.com/pytorch/pytorch  # 不止git pytorch仓库下的文件，还有所依赖的库
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
```

创建conda环境 ，在环境中来build

```shell
# conda 语句，-n为命名， -y为自动确认
conda create -y -n torch-build python=3.10 # 创建环境并命名为torch-build
conda activate torch-build    # 激活并进入这个环境；退出环境为 conda deactivate
```

此时在终端中应该出现环境名，例如 

```shell
(torch-build) root@my_machine:/pytorch#
```

在激活后的环境中pip换源到清华，[Tuna pypi源](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)

```shell
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

```

## 3 - 安装编译必要的依赖

```shell
cd pytorch
conda install cmake ninja
pip install -r requirements.txt
pip install mkl-static mkl-include
# CUDA only: Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda124  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo 之前确定的版本是cuda12.4所以这里 magma-cuda124

# (optional) If using torch.compile with inductor/triton, install the matching version of triton
# Run from the pytorch directory after cloning
# For Intel GPU support, please explicitly `export USE_XPU=1` before running command.
make triton
```

## 4 - 安装PyTorch

编译的过程可能会较长，建议挂[Tmux](https://www.ruanyifeng.com/blog/2019/10/tmux.html#:~:text=1.2%20Tmux%20%E7%9A%84)来执行，以免因为网络掉线而失去终端的信息

```shell
export _GLIBCXX_USE_CXX11_ABI=1 # 启用new C++ABI
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
# 执行下面语句的时候，应该在 pytorch/ 目录下

#!!!!!编译前务必使用htop检查CPU占用率，以免造成服务器崩掉

MAX_JOBS=8 python setup.py develop    
# 此过程可能出错，需安装 pip install pyyaml typing_extensions 
# 可以限制一下线程数量以免爆内存！！！

#如果遇到找不到cuda的问题请执行
MAX_JOBS=8 CMAKE_ARGS="-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.4" python setup.py develop

```

## 5 - 验证安装是否正确

```shell
... 
Using /root/anaconda3/envs/torch-build/lib/python3.10/site-packages
Finished processing dependencies for torch==2.6.0a0+gitd6f340f  # 编译正确的输出

(torch-build) root@my_machine:/pytorch# python
Python 3.10.15 (main, Oct  3 2024, 07:27:34) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.__version__
'2.6.0a0+gitd6f340f'

#如果GLBCXX 3.4.30 not found
cd /home/parallel_X/anaconda3/torch-build/lib/ #根据自己的路径修改
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
```



# 代码模板介绍

+ 代码参考 [李沐 等 《动手学机器学习》](https://zh-v2.d2l.ai/chapter_attention-mechanisms/transformer.html) ，样例整理了其中PyTorch实现的代码，并组织为了较容易理解的形式。
+ 为简化问题，将数据集一并发送给同学们。如本地不存在该数据集，` data_preprocess.py`和`test_preprocess_text.py`也有对应的代码可以现场下载到文件夹`/data`下（需要梯子）。
+ `Lab4-tamplate`文件结构介绍

```shell
Lab4-tamplate
├── log   # 训练的loss图，推理4个例子的heatmap
├── data  # 存放NLP任务的数据集 
│   ├── fra-eng
│   │   ├── _about.txt
│   │   └── fra.txt
│   ├── fra-eng.zip  # Transformer训练任务的数据集：英-法翻译
│   └── timemachine.txt
├── data_preprocess.py       # 数据预处理文件
├── test_preprocess_text.py  # 测试网络 是否可下载数据集
├── transformer_main.py      # Transformer 主函数
├── transformer_module.py    # 功能模块，可观察encoder-decoder结构
└── utils
    ├── matplotlib_draw.py   # /log 中的作图代码
    └── timer.py             # 计时函数
```

+ 使用`nvidia-smi`命令查看GPU使用率
+ 执行 `transformer_main.py` 会首先调用 `data_preprocess.py` 处理好数据集，然后通过函数模块 `transformer_module.py` 实现主要功能（即可以与论文对上的模型结构），训练和推理结果通过`matplotlib_draw.py`作图可视化。
+ 找不到matplotlib执行`conda install matplotlib`

