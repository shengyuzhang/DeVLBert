import os
import numpy as np
# from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ
from tensorpack.dataflow import *
import lmdb
import json
import pdb
import csv
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'cls_prob']
import sys
import pandas as pd
import zlib
import base64

csv.field_size_limit(sys.maxsize)


def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', names=["caption","url"], usecols=range(0,2))
    df['folder'] = folder
    print("Processing", len(df), " Images:")
    return df
# def _file_name(row):
#     return "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))

class Conceptual_Caption(RNGDataFlow):
    """
    """
    def __init__(self, corpus_path_1, corpus_path_2, shuffle=False):
        """
        Same as in :class:`ILSVRC12`.
        """
        self.shuffle = shuffle
        self.num_file_1 = 16
        self.name_1 = os.path.join(corpus_path_1, 'CC_resnet101_faster_rcnn_genome.tsv.%d')
        self.num_file_2 = 16
        self.name_2 = os.path.join(corpus_path_2, 'sbu_resnet101_faster_rcnn_genome.tsv.%d')        
        self.infiles = [self.name_1 % i for i in range(self.num_file_1)] + [self.name_2 % i for i in range(self.num_file_2)]
        self.counts = []
        self.num_caps = 3103920 + 874544 # TODO: compute the number of processed pics

        self.captions = {}
        df = open_tsv('/mnt/yangan.ya/VL-BERT/data/conceptual-captions/utils/Train_GCC-training.tsv', 'training')
        for i, img in enumerate(df.iterrows()):
            caption = img[1]['caption']#.decode("utf8")
            url = img[1]['url']
            # im_name = _file_name(img[1])
            # image_id = im_name.split('/')[1]
            image_id = str(i)
            self.captions[image_id] = caption
        
        with open('/mnt/yangan.ya/VL-BERT/data/conceptual-captions/utils/SBU_captioned_photo_dataset_captions.txt') as finsbu:
            offset = 3318333
            sbu_captions = [line.strip() for line in finsbu]
            for i in range(len(sbu_captions)):
                self.captions[str(i + offset)] = sbu_captions[i]
        
        json.dump(self.captions, open('/mnt3/yangan.ya/features_lmdb/CCsbu/caption_train.json', 'w'))

    def __len__(self):
        return self.num_caps

    def __iter__(self):
        for infile in self.infiles:
            # count = 0
            with open(infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                for item in reader:
                    image_id = item['image_id']
                    image_h = item['image_h']
                    image_w = item['image_w']
                    num_boxes = item['num_boxes']
                    boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(int(num_boxes), 4)
                    features = np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape(int(num_boxes), 2048)
                    cls_prob = np.frombuffer(base64.b64decode(item['cls_prob']), dtype=np.float32).reshape(int(num_boxes), 1601)
                    caption = self.captions[image_id]

                    yield [features, cls_prob, boxes, num_boxes, image_h, image_w, image_id, caption]

if __name__ == '__main__':
    corpus_path_1 = '/mnt3/yangan.ya/features/CC'
    corpus_path_2 = '/mnt3/yangan.ya/features/sbu'
    ds = Conceptual_Caption(corpus_path_1, corpus_path_2)
    ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    LMDBSerializer.save(ds1, '/mnt3/yangan.ya/features_lmdb/CCsbu/training_feat_all.lmdb')
