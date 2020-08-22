# first step
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 train_concap.py \
--from_pretrained /mnt/xuesheng_1/bert-base-uncased \
--bert_model /mnt/xuesheng_1/bert-base-uncased --config_file config/bert_base_6layer_6conect.json \
--learning_rate 1e-4 --train_batch_size 64 \
--save_name devlbert_base --distributed

# second step
# change region mask probability from 0.15 to 0.3
# i.e. modify ./devlbert/dataset/concept_cap_dataset.py line 579, 580  0.15->0.3

## third step
#python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 train_concap.py \
#--from_pretrained save/devlbert_base/pytorch_model_11.bin \
#--bert_model /mnt/xuesheng_1/bert-base-uncased --config_file config/bert_base_6layer_6conect.json \
#--learning_rate 1e-4 --train_batch_size 64 \
#--save_name devlbert --distributed