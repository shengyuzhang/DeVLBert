CUDA_VISIBLE_DEVICES=0 python eval_tasks.py \
--bert_model /mnt/xuesheng_1/bert-base-uncased --from_pretrained save/VQA_bert_base_6layer_6conect-devlbert_vqa/vqa-pytorch_model_13_ema.bin \
--config_file config/bert_base_6layer_6conect.json \
--task 0 --split test --save_name devlbert_vqa
