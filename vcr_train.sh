python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 train_vcr.py \
--bert_model /mnt/xuesheng_1/bert-base-uncased --from_pretrained save/devlbert/pytorch_model_11.bin  \
--config_file config/bert_base_6layer_6conect.json  --learning_rate 2e-5 --num_workers 16 \
--tasks 1-2 --save_name devlbert_vcr --num_train_epochs 10 \
--use_ema --ema_decay_ratio 0.9998