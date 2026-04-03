import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml


def ensure_dir(path):
    #确保目录存在
    
    path =Path(path)

    path.mkdir(parents=True, exist_ok=True)

def set_seed(seed:int)->None:
    #设定种子，提升可复现性 种子相同，随机序列相同；种子不同，随机序列通常不同。

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # 下面两行是为了让 cuDNN 的行为更稳定、更可复现
    # deterministic=True：尽量使用确定性算法
    torch.backends.cudnn.deterministic = True

    # benchmark=False：关闭自动寻找最快算法
    # 因为“最快”有时会带来不完全可复现
    torch.backends.cudnn.benchmark = False    


def seed_worker(worker_id:int) ->None: #开多个worker读取数据的时候，给美国worker单独设种子
     # % (2 ** 32) 是为了把它限制到 numpy 能安全使用的整数范围    无论 PyTorch 给的种子多大，都把它变成一个 NumPy 能稳定使用的 32 位范围内整数
    worker_seed = torch.initial_seed() %(2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def read_csv_rows(path):
    """
    读取csv文件，把csv文件的每一行变成一个字典
    读出来的一行:
        {"image_name": "0001.jpg", "video_id": "5"}
    返回一个列表，列表的每一个元素都是一个字调
    """
    with open(path, "r", encoding= "utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        return list(reader)
    
def write_csv_rows(path, rows, fieldnames) ->None:
    """
    把字典列表写进csv里
    """
    path =Path(path)

    ensure_dir(path.parent)

    with open(path,"w", encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader() #先写表头
        writer.writerows(rows) #再把所有数据写进去

def read_json(path):
    """ 
    读取一个json文件，并转成Python对象
    """

    with open(path,"r", encoding="utf-8") as f:
        #从文件f中读取json内容，并转成Python 数据
        return json.load(f)

def save_json(path,obj)->None:

    path= Path(path)
    ensure_dir(path.parent)

    with open(path,"w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_yaml(path):
    """
    读取 yaml 配置文件。读出来后通常会得到一个字典。
    """

    with open(path, "r", encoding="utf-8") as f:
        # yaml.safe_load 是安全读取 yaml 的常用方法
        data = yaml.safe_load(f)

    # 有些空 yaml 文件会读出 None
    # 所以这里做一个保护：
    # 如果 data 是 None，就返回一个空字典 {}
    if data is None:
        return {}

    return data

