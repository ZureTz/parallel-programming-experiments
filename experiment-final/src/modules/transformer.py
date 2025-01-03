import math
from typing import Any

import torch
from torch import nn, Tensor


def transpose_qkv(X: Tensor, num_heads: int) -> Tensor:
    """
    为了多注意力头的并行计算而变换形状

    Args:
        X (Tensor): 经过变换的形状为(batch_size, 查询或者“键－值”对的个数, num_hiddens)
        num_heads (int): 多注意力头的数量
    """
    # 输入X的形状:(batch_size, 查询或者“键－值”对的个数, num_hiddens)
    # 输出X的形状:(batch_size, 查询或者“键－值”对的个数, num_heads, num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size, num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:
    # (batch_size*num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X: Tensor, num_heads: int) -> Tensor:
    """
    逆转transpose_qkv函数的操作

    Args:
        X (Tensor): 经过变换的形状为(batch_size, 查询的个数, num_hiddens)
        num_heads (int): 多注意力头的数量

    Returns:
        Tensor: 逆转操作后的张量
    """
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


def sequence_mask(X: Tensor, valid_len: Tensor, value: int = 0) -> Tensor:
    """
    在序列中屏蔽不相关的项

    Args:
        X (Tensor): 3D张量
        valid_len (Tensor): 1D或2D张量
        value (int, optional): 要替换的值. 默认为 0.

    Returns:
        Tensor: 一个与X形状相同的张量, 其中有效长度之后的元素被替换为value
    """
    maxlen = X.size(1)
    mask = (
        torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]
        < valid_len[:, None]
    )
    X[~mask] = value
    return X


def masked_softmax(X: Tensor, valid_lens: Tensor) -> Tensor:
    """
    通过在最后一个轴上掩蔽元素来执行softmax操作

    Args:
        X (Tensor): 3D张量
        valid_lens (Tensor): 1D或2D张量

    Returns:
        Tensor: 通过softmax操作后的张量
    """
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class Encoder(nn.Module):
    """
    Encoder-Decoder架构的基本编码器接口

    Args:
        nn.Module: PyTorch的神经网络模块
    """

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X: Tensor, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """
    编码器-解码器架构的基本解码器接口

    Args:
        nn.Module: PyTorch的神经网络模块
    """

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class AttentionDecoder(Decoder):
    """
    带有注意力机制解码器的基本接口

    Args:
        Decoder: 编码器-解码器架构的基本解码器接口
    """

    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """
    编码器-解码器架构的基类

    Args:
        nn.Module: PyTorch的神经网络模块
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """
    带遮蔽的softmax交叉熵损失函数

    Args:
        nn.CrossEntropyLoss: PyTorch的交叉熵损失函数
    """

    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred: Tensor, label: Tensor, valid_len: Tensor) -> Tensor:
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = "none"
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label
        )
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


class Accumulator:
    """
    在n个变量上累加。
    """

    def __init__(self, n: int):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]


class PositionalEncoding(nn.Module):
    """
    位置编码

    Args:
        nn.Module: PyTorch的神经网络模块
    """

    def __init__(self, num_hiddens: int, dropout: float, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X: Tensor) -> Tensor:
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


class DotProductAttention(nn.Module):
    """
    缩放点积注意力

    Args:
        nn.Module: PyTorch的神经网络模块
    """

    def __init__(self, dropout: float, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(
        self, queries: Tensor, keys: Tensor, values: Tensor, valid_lens: Tensor = None
    ) -> Tensor:
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度、

        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """
    多头注意力

    Args:
        nn.Module: PyTorch的神经网络模块
    """

    def __init__(
        self,
        key_size: int,
        query_size: int,
        value_size: int,
        num_hiddens: int,
        num_heads: int,
        dropout: float,
        bias: bool = False,
        **kwargs: Any
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)

        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(
        self, queries: Tensor, keys: Tensor, values: Tensor, valid_lens: Tensor
    ) -> nn.Linear:
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionWiseFFN(nn.Module):
    """
    基于位置的前馈网络

    Args:
        nn.Module: PyTorch的神经网络模块
    """

    def __init__(
        self, ffn_num_input: int, ffn_num_hiddens: int, ffn_num_outputs: int, **kwargs
    ):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X) -> nn.Linear:
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """
    残差连接后进行层规范化

    Args:
        nn.Module: PyTorch的神经网络模块
    """

    def __init__(self, normalized_shape: list[int], dropout: float, **kwargs: Any):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y) -> nn.LayerNorm:
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    """
    Transformer编码器块

    Args:
        nn.Module: PyTorch的神经网络模块
    """

    def __init__(
        self,
        key_size: int,
        query_size: int,
        value_size: int,
        num_hiddens: int,
        norm_shape: list[int],
        ffn_num_input: int,
        ffn_num_hiddens: int,
        num_heads: int,
        dropout: float,
        use_bias: bool = False,
        **kwargs: Any
    ):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(Encoder):
    """
    Transformer编码器

    Args:
        Encoder: 编码器-解码器架构的基本编码器接口
    """

    def __init__(
        self,
        vocab_size: int,
        key_size: int,
        query_size: int,
        value_size: int,
        num_hiddens: int,
        norm_shape: list[int],
        ffn_num_input: int,
        ffn_num_hiddens: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        use_bias: bool = False,
        **kwargs: Any
    ):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                EncoderBlock(
                    key_size,
                    query_size,
                    value_size,
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    use_bias,
                ),
            )

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):
    """
    解码器中第i个块

    Args:
        nn.Module: PyTorch的神经网络模块
    """

    def __init__(
        self,
        key_size: int,
        query_size: int,
        value_size: int,
        num_hiddens: int,
        norm_shape: list[int],
        ffn_num_input: int,
        ffn_num_hiddens: int,
        num_heads: int,
        dropout: float,
        i: int,
        **kwargs: Any
    ):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(
                batch_size, 1
            )
        else:
            dec_valid_lens = None

        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(AttentionDecoder):
    """
    Transformer解码器

    Args:
        AttentionDecoder: 带有注意力机制解码器的基本接口
    """

    def __init__(
        self,
        vocab_size: int,
        key_size: int,
        query_size: int,
        value_size: int,
        num_hiddens: int,
        norm_shape: list[int],
        ffn_num_input: int,
        ffn_num_hiddens: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        **kwargs: Any
    ):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                DecoderBlock(
                    key_size,
                    query_size,
                    value_size,
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    i,
                ),
            )
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
