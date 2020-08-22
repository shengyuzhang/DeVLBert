CUDA_VISIBLE_DEVICES=1 python eval_retrieval.py \
--bert_model /mnt/xuesheng_1/bert-base-uncased \
--from_pretrained save/RetrievalFlickr30k_bert_base_6layer_6conect-devlbert_ir/pytorch_model_11_ema.bin \
--config_file config/bert_base_6layer_6conect.json \
--task 3 --split test --batch_size 1