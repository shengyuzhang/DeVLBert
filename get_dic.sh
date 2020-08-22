python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 get_dic.py \
--from_pretrained /mnt/xuesheng_1/bert-base-uncased \
--bert_model /mnt/xuesheng_1/bert-base-uncased --config_file config/bert_base_6layer_6conect.json \
--learning_rate 1e-4 --train_batch_size 16 \
--save_name causal_pretrained --distributed