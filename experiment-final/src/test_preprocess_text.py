# Description: This file is used to test the preprocess_text.py file.
# Path: experiment-final/src/test_preprocess_text.py

from ast import main
import re
import os
import hashlib
import requests

# URL to download the dataset
DATA_URL: str = "http://d2l-data.s3-accelerate.amazonaws.com/"

# Dictionary to store the dataset
DATA_HUB: dict = dict()
DATA_HUB["time_machine"] = (
    DATA_URL + "timemachine.txt",
    "090b5e7e70c295757f55df93cb0a180b9691891a",
)


def download(name: str, cache_dir: str = os.path.join(".", "data")) -> str:
    """
    下载一个DATA_HUB中的文件, 返回本地文件名

    Args:
        name (str): 在DATA_HUB中下载的文件名
        cache_dir (str, optional): 存储下载文件的本地目录. 默认为 os.path.join(".", "data").
    
    Returns:
        str: 下载文件的本地文件名
    """

    # Check if the file is in the DATA_HUB
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    # Get the URL and SHA-1 hash value from the DATA_HUB
    url, sha1_hash = DATA_HUB[name]

    # Create the cache directory if it does not exist
    # And create the file name
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split("/")[-1])

    # Check if the file exists
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


def read_time_machine() -> list[str]:  # @save
    """
    将时间机器数据集加载到文本行的列表中

    Returns:
        list[str]: 时间机器数据集的文本行列表
    """

    with open(download("time_machine"), "r") as f:
        lines = f.readlines()

    # Return the list of text lines after removing special characters and converting to lowercase
    return [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines]


def main():
    lines = read_time_machine()
    print(f"# 文本总行数: {len(lines)}")
    print(lines[0])
    print(lines[10])


if __name__ == "__main__":
    main()
