# DeVLBert: Learning Deconfounded Visio-Linguistic Representations

Original implementation for paper [DeVLBert: Learning Deconfounded Visio-Linguistic Representations](https://arxiv.org/abs/2008.06884).

## Repository Setup

1. Create a fresh conda environment, and install all dependencies.

```sh
conda create -n devlbert python=3.6
conda activate devlbert
git clone https://github.com/shengyuzhang/DeVLBert.git
cd devlbert
pip install -r requirements.txt
```

2. Install pytorch

```sh
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

3. Install apex, follows https://github.com/NVIDIA/apex

4. compile tools

```sh
cd tools/refer
make
```

## Data Setup

Check `README.md` under `data` for more details.  Check  `devlbert_tasks.yml` for more details.

We totally follow the setup of [vilbert](https://github.com/jiasenlu/vilbert_beta).

## Get DeVLBert pre-trained model

This repo is only for design D in our paper. You can realize other designs easily based on the repo.

### Pre-trained model for Evaluation

You can download our pre-trained DeVLBert model in [here](https://drive.google.com/file/d/151vQVATAlFM6rs5qjONMnIJBGfL8ea-B/view?usp=sharing) and put it under `save/devlbert/`.

### Train DeVLBert model by yourself

1: Follow [Data Setup](#Data-Setup) and get training dataset. Download pretrain bert-base-uncased model in [here](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz) and bert-base-uncased vocabulary in [here](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt).

2: Run `./dic/get_noun_set.py` `./dic/count.py`  `./dic/get_id2class.py` in order to get `"./dic/id2class.npy"`. Run `get_dic.sh` and `./dic/merge_dic.ipynb` to get confounder dictionaries.

Absolute paths often occur in our code, the meaning is as follows:

- `"/mnt3/xuesheng/features_lmdb/CC/training_feat_part_" + str(rank) + ".lmdb"` : We process the Concept Caption dataset and divide it into 8 segments
- `"/mnt3/xuesheng/features_lmdb/CC/caption_train.json"` : During the processing the Concept Caption dataset, we save all captions. It will be used in the process of training, because we need to get a wrong caption of a image for visio-linguistic alignment proxy task.
- `"/mnt/xuesheng_1/bert-base-uncased"` : We put bert-base-uncased model and vocabulary in here.
- `"./dic/id2class.npy"` `"./dic/id2class1155.npy"` : We keep every sentence 2 confound words and 4 confound words to get the former and the latter, respectively.

3: Follow `train.sh`. Firstly, run `train.sh`. Secondly, change region mask probability from 0.15 to 0.3. Thirdly, run `train.sh` again. We totally train 24(12 + 12) epochs. You can train for longer time for higher performance, especially in **Zero-Shot Image Retrieval** task.

## Evaluation

Follow [Data Setup](#Data-Setup) and get all datasets. Please note that keep the dataset path consistent with the path in `devlbert_tasks.yml`.

### VQA

1: Finetune: Run `vqa_train.sh`. Or you can directly download our trained model in [here](https://drive.google.com/file/d/1bZzr47lbqNALn_OynodpjEycdwGyAuC-/view?usp=sharing).

2: Inference: Modify `devlbert_tasks.yml`: comment line 7 and uncoment line 8. Then run `vqa_test.sh`. The result will be generated at `results/VQA_bert_base_6layer_6conect-{save_name of vqa_train.sh}-{save_name of vqa_test.sh}/test_result.json`.

3: Evaluation: Access [VQA Challenge 2020](https://evalai.cloudcv.org/web/challenges/challenge-page/514/overview) and sign up for an account. Submit your result in **Test-Dev Phase** or **Test-Standard Phase**.

### VCR

We only evaluate on the validation set. Run `vcr_train.sh`, and you can get result at the first several lines of `save/VCR_Q-A-VCR_QA-R_bert_base_6layer_6conect-{save_name of vcr_train.sh}/output.txt`.

### Image Retrieval

1: Finetune: Run `ir_train.sh`. Or you can directly download our trained model in [here](https://drive.google.com/file/d/1B0v7rWjzOITlDyuaypbcUZ8boVAn7N_Q/view?usp=sharing).

2: Evaluation: Modify `devlbert_tasks.yml`: comment line 60,64 and uncoment line 61,65. Then run `ir_test.sh`. The result will be printed on the screen after evaluation finishing.

### Zero-Shot Image Retrieval

Run `zsir_test.sh` directly, and the result will be printed on the screen after evaluation finishing.

### RefCOCO+

We only evaluate on the validation set. Run `refcoco_train.sh`, and you can get result at the first several lines of `save/refcoco+_bert_base_6layer_6conect-{save_name of refcoco_train.sh}/output.txt`.

## References

If you use DeVLBert in your research or wish to refer to the results, please cite our paper

~~~
@article{zhang2020devlbert,
  title={DeVLBert: Learning Deconfounded Visio-Linguistic Representations},
  author={Zhang, Shengyu and Jiang, Tan and Wang, Tan and Kuang, Kun and Zhao, Zhou and Zhu, Jianke and Yu, Jin and Yang, Hongxia and Wu, Fei},
  journal={arXiv preprint arXiv:2008.06884},
  year={2020}
}
~~~