import json
import time
import argparse

import numpy as np
from keras.models import load_model
from keras_bert import get_custom_objects

from model_train import token_dict, OurTokenizer

parser = argparse.ArgumentParser(description='ArgparseTry')
parser.add_argument('--path',required=True,type=str)
args = parser.parse_args()


# -*- coding: utf-8 -*-
# @Time : 2020/12/23 15:28
# @Author : Jclian91
# @File : model_predict.py
# @Place : Yangpu, Shanghai
# 模型預測腳本



maxlen = 300

# Load訓練好的模型
model = load_model("cls_cnews.h5", custom_objects=get_custom_objects())
tokenizer = OurTokenizer(token_dict)
with open("label.json", "r", encoding="utf-8") as f:
        label_dict = json.loads(f.read())

s_time = time.time()
# 預測例句
def readfile(path):
    with open(path,'r',encoding='utf-8-sig') as f:
        content=f.readlines()
        f.close()
    return content

text = readfile(args.path)
ntext=str(text).replace("'",'')

    # "說到硬派越野SUV，你會想起哪些車型？是被稱為“霸道”的豐田 普拉多 (配置 | 詢價) ，還是被叫做“山貓”的帕傑羅，亦或者是“渣男專車”奔馳大G、" \
    # "“沙漠王子”途樂。總之，隨著世界各國越來越重視對環境的保護，那些大排量的越野SUV在不久的將來也會漸漸消失在我們的視線之中，所以與其錯過，" \
    # "不如趁著還年輕，在有生之年裡趕緊去入手一台能讓你心儀的硬派越野SUV。而今天我想要來跟你們聊的，正是全球公認的十大硬派越野SUV，" \
    # "越野迷們看完之後也不妨思考一下，到底哪款才是你的菜，下面話不多說，趕緊開始吧。"

# 利用BERT進行tokenize
ntext = ntext[:maxlen]
X1, X2 = tokenizer.encode(first=ntext, max_len=maxlen)
# 模型預測並輸出預測結果
predicted = model.predict([[X1], [X2]])
y = np.argmax(predicted[0])
print("原文: %s" % ntext)
print("預測標籤: %s" % label_dict[str(y)])
e_time = time.time()
print("cost time:", e_time - s_time)


