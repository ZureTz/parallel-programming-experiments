import random
from typing import Any, Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from modules.data_preprocess import Vocab, bleu, load_data_nmt, truncate_pad
from modules.transformer import (
    Accumulator,
    EncoderDecoder,
    TransformerEncoder,
    TransformerDecoder,
    MaskedSoftmaxCELoss,
)

from utils.timer import Timer, time_stamp_cudasync
from utils.matplotlib_draw import Animator, draw_roofline, show_heatmaps


def try_gpu(i: int = 0) -> torch.device:
    """
    如果存在, 则返回gpu(i), 否则返回cpu()

    Args:
        i (int, optional): GPU的索引. 默认为 0.

    Returns:
        torch.device: GPU(i) 或 CPU()
    """

    if torch.cuda.device_count() < i + 1:
        print("CUDA is not available !")
        return torch.device("cpu")

    print("Pytorch version\t:", torch.__version__)
    print("CUDA version\t:", torch.version.cuda)
    print("GPU\t\t:", torch.cuda.get_device_name(), "\n", "-" * 50)
    return torch.device(f"cuda:{i}")


def grad_clipping(net: nn.Module | Any, theta: int):
    """
    裁剪梯度

    Args:
        net (nn.Module): 神经网络
        theta (int): 裁剪的阈值
    """
    if not isinstance(net, nn.Module):
        return
        
    # 使用PyTorch内置的梯度裁剪函数，更高效且数值稳定
    torch.nn.utils.clip_grad_norm_(
        parameters=net.parameters(),
        max_norm=theta,
        norm_type=2.0
    )


def train_seq2seq(
    net: EncoderDecoder,
    data_iter: DataLoader[Tuple[Tensor, ...]],
    lr: float,
    num_epochs: int,
    tgt_vocab: Vocab,
    device: torch.device,
):
    """
    训练序列到序列模型

    Args:
        net (EncoderDecoder): Transformer模型
        data_iter (DataLoader): 训练数据迭代器
        lr (float): 学习率
        num_epochs (int): 训练迭代次数
        tgt_vocab (Vocab): 目标语言词汇表
        device (torch.device): 训练设备
    """

    def xavier_init_weights(m: nn.Linear | nn.GRU):
        """
        初始化权重

        Args:
            m (nn.Linear | nn.GRU): 线性层或GRU层
        """
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = Animator(xlabel="epoch", ylabel="loss", xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)  # 训练损失总和, 词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor(
                [tgt_vocab["<bos>"]] * Y.shape[0], device=device
            ).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # 损失函数的标量进行“反向传播”
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))

    animator.save()
    print(
        f"\nloss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} "
        f"tokens/sec on {str(device)}"
    )


def predict_seq2seq(
    net: EncoderDecoder,
    src_sentence: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    num_steps: int,
    device: torch.device,
    save_attention_weights: bool = False,
) -> Tuple[str, list[Tensor]]:
    """
    序列到序列模型的预测

    Args:
        net (EncoderDecoder): Transformer模型
        src_sentence (str): 输入语句
        src_vocab (Vocab): 源语言词汇表
        tgt_vocab (Vocab): 目标语言词汇表
        num_steps (int): 解码器的最大步数
        device (torch.device): 训练设备
        save_attention_weights (bool, optional): 是否保存注意力权重. 默认为 False.

    Returns:
        Tuple[str, list[Tensor]]: 预测的目标语言句子, 注意力权重列表
    """

    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(" ")] + [src_vocab["<eos>"]]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab["<pad>"])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0
    )
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(
        torch.tensor([tgt_vocab["<bos>"]], dtype=torch.long, device=device), dim=0
    )
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元, 作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测, 输出序列的生成就完成了
        if pred == tgt_vocab["<eos>"]:
            break
        output_seq.append(pred)
    return " ".join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def main():
    # 决定模型尺寸的参数, 一般不做调整
    num_hiddens, num_layers, dropout, ffn_num_input, ffn_num_hiddens, num_heads = (
        32,
        2,
        0.1,
        32,
        64,
        4,
    )
    # 机器学习模型训练相关参数
    lr, num_epochs, num_steps, device = 0.005, 200, 10, try_gpu()

    # 观察性能 可以调整的参数
    batch_size, seq_len = 64, 32
    running_iters, warmup_iters = 100, 20
    key_size, query_size, value_size = seq_len, seq_len, seq_len
    norm_shape = [32]

    # 设置 float32 矩阵乘法的精度
    torch.set_float32_matmul_precision("high")

    # 数据预处理, 包含了了词向量化等步骤
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)

    # 实例化 Transformer 的关键组件

    encoder = TransformerEncoder(
        len(src_vocab),
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
    )

    decoder = TransformerDecoder(
        len(tgt_vocab),
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
    )

    net = EncoderDecoder(encoder, decoder)

    # 编译模型以提高性能
    net = torch.compile(net)

    # 训练模型
    time_start = time_stamp_cudasync()
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
    time_end = time_stamp_cudasync()
    train_time_per_epoch = (time_end - time_start) * 1000 / num_epochs
    print("Train Speed: {:.3f} ms / epoch".format(train_time_per_epoch))

    # 预定义的英语句子列表
    eng_sentences = [
        "go .",
        "i lost .",
        "he's calm .",
        "i'm home .",
        "she is beautiful .",
        "they are happy .",
        "we will win .",
        "you are welcome .",
        "it is raining .",
        "the sky is blue .",
    ]
    random_engs = random.choices(eng_sentences, k=running_iters + warmup_iters)

    # 进行 running_iters + warmup_iters 次模型推理
    time_start = time_stamp_cudasync()
    for i in range(running_iters + warmup_iters):
        if i == warmup_iters:
            time_start = time_stamp_cudasync()
        eng = random_engs[i]
        translation, dec_attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True
        )
    time_end = time_stamp_cudasync()
    infer_time_per_iter = (time_end - time_start) * 1000 / running_iters
    print("Infer Speed: {:.3f} ms / iter".format(infer_time_per_iter))

    # 进行4次模型推理, 完成英语到法语的翻译任务 --- 演示翻译效果
    # 检查模型功能是否正确：功能正确则至少出现一个bleu 1.000；否则错误
    print("\n4 example of  English => French  translation task")
    engs = ["go .", "i lost .", "he 's calm .", "i 'm home ."]
    fras = ["va !", "j 'ai perdu .", "il est calme .", "je suis chez moi ."]
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True
        )
        print(f"{eng} => {translation}, ", f"bleu {bleu(translation, fra, k=2):.3f}")

    # 可视化注意力权重
    enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape(
        (num_layers, num_heads, -1, num_steps)
    )

    # 显示注意力权重
    show_heatmaps(
        enc_attention_weights.cpu(),
        xlabel="Key positions",
        ylabel="Query positions",
        titles=["Head %d" % i for i in range(1, 5)],
        figsize=(7, 3.5),
    )

    draw_roofline(batch_size, seq_len, num_hiddens, num_heads)


if __name__ == "__main__":
    main()
