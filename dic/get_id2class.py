import lmdb
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer
import json
import tensorpack.dataflow as td
import collections

tokenizer = BertTokenizer.from_pretrained(
    "/mnt/xuesheng_1/bert-base-uncased", do_lower_case=True
)

with open('noun_frequency.json', 'r') as f:
    dic = json.load(f)

dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
d = collections.OrderedDict()
for tup in dic:
    d[tup[0]] = tup[1]

# process noun_frequency, guarantee that every sentence have 4 confound words
new_dic = {}
cur = 0
# 387991 is the number of samples in a dataset segment, 8 segments
limit = 387991 * 8 * 4
for k, v in d.items():
    new_dic[k] = v
    cur += v
    if cur >= limit:
        break


# change token to bert id, and change bert id to class
noun_ids = []
for key in new_dic:
    noun_ids.append(tokenizer.vocab[key])

d = {}
for i in range(len(noun_ids)):
    d[noun_ids[i]] = i

print("Noun vocabulary size is {}".format(len(d)))
np.save("./id2class1155.npy", d)

