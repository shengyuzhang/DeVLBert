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
    ds = td.LMDBSerializer.load("/mnt3/xuesheng/features_lmdb/CC/training_feat_part_%d.lmdb"%num, shuffle=False)
    num_dataset = len(ds)
    print(num_dataset)
    ds.reset_state()
    se = set()
    nonu = ["NN", "NNS", "NNP", "NNPS"]
    for i, batch in enumerate(ds.get_data(), 1):
        image_feature_wp, image_target_wp, image_location_wp, num_boxes, image_h, image_w, image_id, caption = batch
        tokens = tokenizer.tokenize(caption)
        while len(tokens) > 34:
            tokens.pop()
        tag_tokens = nltk.pos_tag(tokens)
        l = len(tag_tokens)
        assert l <= 34
        for j in range(l):
            if tag_tokens[j][1] in nonu and tag_tokens[j][0][0] != "#" and (j != l-1 and tag_tokens[j+1][0][0] != "#" or j == l-1):
                # dic[tag_tokens[j][0]] = dic.get(tag_tokens[j][0], 0) + 1
                se.add(tag_tokens[j][0])
        if i % 2000 == 0:
            print("process {} has done {}".format(num, i))

    q.put(se)
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
        se = q.get()
        arr.append(se)

    for process in pool:
        process.join()
    print("join Done")

    res = set()
    for se in arr:
        res = res | se

    with open('noun_set.json', 'w') as f:
        json.dump(list(res), f, indent=4)