CUDA_VISIBLE_DEVICES=0 python eval_retrieval.py \
--bert_model /mnt/xuesheng_1/bert-base-uncased \
--from_pretrained save/devlbert/pytorch_model_11.bin \
--config_file config/bert_base_6layer_6conect.json \
--task 3 --split test --batch_size 1 --zero_shot