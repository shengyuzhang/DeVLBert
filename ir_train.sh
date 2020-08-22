python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port 88888 train_tasks.py \
--bert_model /mnt/xuesheng_1/bert-base-uncased --from_pretrained save/devlbert/pytorch_model_11.bin  \
--config_file config/bert_base_6layer_6conect.json  --learning_rate 4e-5 --num_workers 9 \
--tasks 3 --save_name devlbert_ir \
--use_ema --ema_decay_ratio 0.9999