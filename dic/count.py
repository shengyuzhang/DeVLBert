import lmdb
import numpy as np
from multiprocessing import Process, Queue
from pytorch_pretrained_bert.tokenization import BertTokenizer
import tensorpack.dataflow as td
import json
import collections
import nltk

def run(q, num):
    tokenizer = BertTokenizer.from_pretrained(
        "/mnt/xuesheng_1/bert-base-uncased", do_lower_case=True
    )
    with open('noun_set.json', 'r') as f:
        se = set(json.load(f))
    ds = td.LMDBSerializer.load("/mnt3/xuesheng/features_lmdb/CC/training_feat_part_%d.lmdb"%num, shuffle=False)
    num_dataset = len(ds)
    print(num_dataset)
    ds.reset_state()
    dic = {}
    for i, batch in enumerate(ds.get_data(), 1):
        image_feature_wp, image_target_wp, image_location_wp, num_boxes, image_h, image_w, image_id, caption = batch
        tokens = tokenizer.tokenize(caption)
        while len(tokens) > 34:
            tokens.pop()
        l = len(tokens)
        assert l <= 34
        for j in range(l):
            if tokens[j] in se and tokens[j][0] != "#" and (j != l-1 and tokens[j+1][0] != "#" or j == l-1):
                dic[tokens[j]] = dic.get(tokens[j], 0) + 1
        if i % 2000 == 0:
            print("process {} has done {}".format(num, i))

    q.put(dic)
    print("finish --- process {}".format(num))

if __name__=='__main__':
    pool = []
    q = Queue()
    for i in range(8):
        process = Process(target=run, args=(q, i))
        pool.append(process)

    for process in pool:
        process.start()
    arr = []
    for i in range(8):
        dic = q.get()
        arr.append(dic)

    for process in pool:
        process.join()
    print("join Done")

    res = {}
    for dic in arr:
        for k, v in dic.items():
            res[k] = res.get(k, 0) + v

    res = sorted(res.items(), key=lambda x: x[1], reverse=True)
    d = collections.OrderedDict()
    for tup in res:
        d[tup[0]] = tup[1]

    with open('noun_frequency.json', 'w') as f:
        json.dump(d, f, indent=4)