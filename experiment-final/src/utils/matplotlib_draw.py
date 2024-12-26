import os
from typing import Final, List, Tuple
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor


class Animator:
    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        legend=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        fmts=("-", "m--", "g-.", "r:"),
        nrows=1,
        ncols=1,
        figsize=(3.5, 2.5),
    ):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [
                self.axes,
            ]
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend
        )
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """设置matplotlib的轴"""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

    def save(self):
        # 确保日志目录存在
        log_dir = "./log"
        os.makedirs(log_dir, exist_ok=True)

        current_time = datetime.now()
        file_name = os.path.join(
            log_dir, f"loss-{current_time.strftime('%Y-%m-%d-%H:%M')}.png"
        )
        plt.savefig(file_name)  # 保存图形到文件名


def show_heatmaps(
    matrices: Tensor,
    xlabel: str,
    ylabel: str,
    titles: List[str] = None,
    figsize: Tuple[float, float] = (2.5, 2.5),
    cmap: str = "Reds",
):
    """
    保存预测的heatmap 到./log下

    Args:
        matrices (Tensor): 3D tensor (batch_size, num_rows, num_cols)
        xlabel (str): x轴标签
        ylabel (str): y轴标签
        titles (List[str], optional): 每个矩阵的标题. Defaults to None.
        figsize (Tuple[float, float], optional): 图形大小. Defaults to (2.5, 2.5).
        cmap (str, optional): 颜色映射. Defaults to "Reds".
    """
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False
    )

    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])

    fig.colorbar(pcm, ax=axes, shrink=0.6)

    current_time = datetime.now()
    file_name = f"./log/heatmap-{current_time.strftime('%Y-%m-%d-%H:%M')}.png"
    plt.savefig(file_name)  # 保存图形到文件名


def attention_flop(
    batch_size: int, seq_len: int, embed_size: int, num_heads: int
) -> float:
    """
    计算 Transformer 中 Attention

    Args:
        batch_size (int): batch size
        seq_len (int): 序列长度
        embed_size (int): 嵌入维度
        num_heads (int): 多头注意力头数

    Returns:
        float: Attention计算的FLOP
    """
    # Q, K, V计算
    flop_per_head = batch_size * seq_len * embed_size * embed_size
    total_flop = flop_per_head * 3 * num_heads

    # Attention得分计算：Q与K的点积
    attention_score_flop = batch_size * num_heads * seq_len * seq_len * embed_size
    total_flop += attention_score_flop

    # softmax和加权计算
    total_flop += attention_score_flop
    return total_flop / 1e9  # 转换为GFLOP


# Roofline模型绘制
def draw_roofline(batch_size: int, seq_len: int, num_hiddens: int, num_heads):
    """
    绘制Roofline模型

    Args:
        batch_size (int): batch size
        seq_len (int): 序列长度
        num_hiddens (int): 隐藏层大小
        num_heads ([type]): 多头注意力头数
    """

    # 定义Roofline模型中的常数值
    theoretical_flop_base: Final[float] = 9465.92  # GFLOP/s 基础时钟理论性能
    theoretical_flop_boost: Final[float] = 12714.68  # GFLOP/s 提升时钟理论性能
    memory_bandwidth: Final[float] = 360  # GB/s 内存带宽

    # 计算 Attention FLOPs
    attention_flops = attention_flop(batch_size, seq_len, num_hiddens, num_heads)

    # 假设不同的 FLOP 数（GFLOP），包括计算出的 FLOP 数
    flops = np.array([attention_flops, 1, 10, 100, 1000])
    op_intensities = flops / memory_bandwidth  # 计算强度 FLOP/Byte

    # 计算性能限制

    # 基础时钟限制的性能
    performance_base = np.minimum(flops, theoretical_flop_base * np.ones_like(flops))
    # 提升时钟限制的性能
    performance_boost = np.minimum(flops, theoretical_flop_boost * np.ones_like(flops))

    # 绘制Roofline模型
    # 设置绘图参数
    plt.figure(figsize=(8, 6))
    # 绘制基础时钟性能
    plt.loglog(
        op_intensities,
        performance_base,
        label="Base Clock Performance",
        linestyle="--",
        marker="o",
    )
    # 绘制提升时钟性能
    plt.loglog(
        op_intensities,
        performance_boost,
        label="Boost Clock Performance",
        linestyle="--",
        marker="x",
    )
    # 绘制内存带宽
    plt.axhline(
        y=memory_bandwidth,
        color="r",
        linestyle="-",
        label="Memory Bandwidth (360 GB/s)",
    )

    plt.xlabel("Operational Intensity (FLOP/Byte)")
    plt.ylabel("Performance (GFLOP/s)")
    plt.title("Attention Mechanism Roofline Model")

    # 设置网格线
    plt.grid(True, which="both", ls="--")
    # 设置图例
    plt.legend()

    # 保存图片
    current_time = datetime.now()
    file_name = (
        f"./log/attention-roofline-model-{current_time.strftime('%Y-%m-%d-%H:%M')}.png"
    )
    plt.savefig(file_name)
