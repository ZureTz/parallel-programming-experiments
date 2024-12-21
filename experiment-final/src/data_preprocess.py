import os
import torch
import tarfile
import zipfile
import hashlib
import requests
import math
import collections

from typing import Tuple
from torch.utils import data

DATA_HUB = dict()
DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
DATA_HUB["fra-eng"] = (
    DATA_URL + "fra-eng.zip",
    "94646ad1522d915e7b0f9296181140edcf86a4f5",
)


def bleu(pred_seq: str, label_seq: str, k: int) -> float:
    # """计算BLEU"""
    """
    Calculate BLEU.

    Args:
        pred_seq (str): The predicted sequence.
        label_seq (str): The label sequence.
        k (int): The maximum length of the n-grams.

    Returns:
        float: The BLEU score.
    """
    pred_tokens, label_tokens = pred_seq.split(" "), label_seq.split(" ")
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[" ".join(label_tokens[i : i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[" ".join(pred_tokens[i : i + n])] > 0:
                num_matches += 1
                label_subs[" ".join(pred_tokens[i : i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def count_corpus(tokens: list) -> collections.Counter:
    """
    Count the frequency of each token in the corpus.

    Args:
        tokens (list): The tokens whose frequencies need to be counted.

    Returns:
        collections.Counter: The frequency of each token in the corpus.
    """

    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def truncate_pad(line: str, num_steps: int, padding_token: str) -> list:
    # """截断或填充文本序列"""
    """
    Truncate or pad text sequences.

    Args:
        line (str): The text sequence.
        num_steps (int): The length of the text sequence.
        padding_token (str): The padding token.

    Returns:
        list: The truncated or padded text sequence.
    """

    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


class Vocab:  # @save
    """
    Vocabulary for text.
    """

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ["<unk>"] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def download(name: str, cache_dir: str = os.path.join(".", "data")) -> str:  # @save
    # """下载一个DATA_HUB中的文件,返回本地文件名"""
    """
    Download a file inserted into DATA_HUB, return the local filename.

    Args:
        name (str): The name of the file to download in DATA_HUB
        cache_dir (str, optional): The local directory to store the downloaded files. Defaults to os.path.join("..", "data").

    Returns:
        str: The local filename of the downloaded file.
    """
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, "rb") as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f"正在从{url}下载{fname}...")
    r = requests.get(url, stream=True, verify=True)
    with open(fname, "wb") as f:
        f.write(r.content)
    return fname


def download_extract(name: str, folder: str = None) -> str:  # @save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == ".zip":
        fp = zipfile.ZipFile(fname, "r")
    elif ext in (".tar", ".gz"):
        fp = tarfile.open(fname, "r")
    else:
        assert False, "只有zip/tar文件可以被解压缩"
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = download_extract("fra-eng")
    with open(os.path.join(data_dir, "fra.txt"), "r", encoding="utf-8") as f:
        return f.read()


def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""

    def no_space(char, prev_char):
        return char in set(",.!?") and prev_char != " "

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace("\u202f", " ").replace("\xa0", " ").lower()
    # 在单词和标点符号之间插入空格
    out = [
        " " + char if i > 0 and no_space(char, text[i - 1]) else char
        for i, char in enumerate(text)
    ]
    return "".join(out)


def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split("\n")):
        if num_examples and i > num_examples:
            break
        parts = line.split("\t")
        if len(parts) == 2:
            source.append(parts[0].split(" "))
            target.append(parts[1].split(" "))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def build_array_nmt(lines, vocab, num_steps):
    # """将机器翻译的文本序列转换成小批量"""
    """
    Convert machine translation text sequences into mini-batches.

    Args:
        lines (_type_): _description_
        vocab (_type_): _description_
        num_steps (_type_): _description_

    Returns:
        _type_: _description_
    """
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab["<eos>"]] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab["<pad>"]) for l in lines])
    valid_len = (array != vocab["<pad>"]).type(torch.int32).sum(1)
    return array, valid_len


def load_array(
    data_arrays: Tuple[torch.tensor], batch_size: int, is_train: bool = True
):  # @save
    """
    Construct a PyTorch data iterator.

    Args:
        data_arrays (Tuple[torch.tensor]): Tuple of data tensors.
        batch_size (int): Batch size.
        is_train (bool, optional): Whether the data is for training. Defaults to True.

    Returns:
        data.DataLoader: Data loader.
    """

    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab
