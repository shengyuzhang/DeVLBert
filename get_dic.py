import argparse
import json
import logging
import os
import random
from io import open
import math
import sys

from time import gmtime, strftime
from timeit import default_timer as timer

import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from devlbert.datasets.concept_cap_dataset2 import ConceptCapLoaderTrain
from devlbert.devlbert2 import BertForMultiModalPreTraining, BertConfig
import torch.distributed as dist

import pdb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_file",
        default="data/conceptual_caption/training",
        type=str,
        # required=True,
        help="The input train corpus.",
    )
    parser.add_argument(
        "--validation_file",
        default="data/conceptual_caption/validation",
        type=str,
        # required=True,
        help="The input train corpus.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default="save",
        type=str,
        # required=True,
        help="The output directory where the model checkpoints will be written.",
    )

    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        # required=True,
        help="The config file which specified the model details.",
    )
    ## Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=36,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument("--predict_feature", action="store_true", help="visual target.")

    parser.add_argument(
        "--train_batch_size",
        default=512,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--img_weight", default=1, type=float, help="weight for image loss"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--on_memory",
        action="store_true",
        help="Whether to load train samples into memory or use disk",
    )
    parser.add_argument(
        "--do_lower_case",
        type=bool,
        default=True,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=3,
        help="Number of workers in the dataloader.",
    )

    parser.add_argument(
        "--save_name",
        default='',
        type=str,
        help="save name for training.",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="Wheter to use the baseline model (single bert)."
    )
    parser.add_argument(
        "--freeze", default = -1, type=int,
        help="till which layer of textual stream of vilbert need to fixed."
    )
    parser.add_argument(
        "--use_chuncks", default=0, type=float, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--distributed", action="store_true" , help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--without_coattention", action="store_true" , help="whether pair loss."
    )
    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="if we need to continue a stopped pretraining procedure, add this"
    )
    args = parser.parse_args()

    print(args)
    if args.save_name is not '':
        timeStamp = args.save_name
    else:
        timeStamp = strftime("%d-%b-%y-%X-%a", gmtime())
        timeStamp += "_{:0>6d}".format(random.randint(0, 10e6))

    savePath = os.path.join(args.output_dir, timeStamp)

    config = BertConfig.from_json_file(args.config_file)

    if args.freeze > config.t_biattention_id[0]:
        config.fixed_t_layer = config.t_biattention_id[0]

    if args.without_coattention:
        config.with_coattention = False
    # save all the hidden parameters.

    bert_weight_name = json.load(open("config/" + "bert-base-uncased_weight_name.json", "r"))
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    special_ids = set()
    for k, v in tokenizer.vocab.items():
        if k[0] == "#":
            special_ids.add(v)

    train_dataset = ConceptCapLoaderTrain(
        args.train_file,
        tokenizer,
        seq_len=args.max_seq_length,
        batch_size=args.train_batch_size,
        predict_feature=args.predict_feature,
        num_workers=args.num_workers,
        distributed=args.distributed,
    )

    # validation_dataset = ConceptCapLoaderVal(
    #     args.validation_file,
    #     tokenizer,
    #     seq_len=args.max_seq_length,
    #     batch_size=args.train_batch_size,
    #     predict_feature=args.predict_feature,
    #     num_workers=2,
    #     distributed=args.distributed,
    # )
    if args.continue_training:
        assert args.start_epoch > 0  # must have pretrained at least one epoch
        num_train_optimization_steps = (
                int(
                    train_dataset.num_dataset
                    / args.train_batch_size
                    / args.gradient_accumulation_steps
                )
                * args.num_train_epochs
        )
    else:
        num_train_optimization_steps = (
                int(
                    train_dataset.num_dataset
                    / args.train_batch_size
                    / args.gradient_accumulation_steps
                )
                * (args.num_train_epochs - args.start_epoch)
        )
    # if args.local_rank != -1:
    #     num_train_optimization_steps = (
    #         num_train_optimization_steps // torch.distributed.get_world_size()
    #     )
    # viz = TBlogger("logs", timeStamp)
    default_gpu = False
    if dist.is_available() and args.distributed:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    # pdb.set_trace()
    if args.predict_feature:
        config.v_target_size = 2048
        config.predict_feature = True
    else:
        config.v_target_size = 1601
        config.predict_feature = False

    if args.from_pretrained:
        if args.continue_training:
            ckpt_load_path = os.path.join(args.from_pretrained, "pytorch_model_{}.bin".format(int(args.start_epoch) - 1))
            model = BertForMultiModalPreTraining.from_pretrained(ckpt_load_path, config)
        else:
            model = BertForMultiModalPreTraining.from_pretrained(args.from_pretrained, config)
    else:
        model = BertForMultiModalPreTraining(config)

    model.cuda()

    if args.fp16:
        model.half()
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if args.freeze != -1:
        bert_weight_name_filtered = []
        for name in bert_weight_name:
            if 'embeddings' in name:
                bert_weight_name_filtered.append(name)
            elif 'encoder' in name:
                layer_num = name.split('.')[2]
                if int(layer_num) <= args.freeze:
                    bert_weight_name_filtered.append(name)

        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if key[12:] in bert_weight_name_filtered:
                value.requires_grad = False

        if default_gpu:
            print("filtered weight")
            print(bert_weight_name_filtered)

    if not args.from_pretrained:
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    else:
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if key[12:] in bert_weight_name:
                    lr = args.learning_rate * 0.1
                else:
                    lr = args.learning_rate

                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.01}
                    ]

                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.0}
                    ]
        if default_gpu:
            print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    # set different parameters for vision branch and lanugage branch.
    if args.fp16:
        try:
            from apex.contrib.optimizers import FP16_Optimizer
            from apex.contrib.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            bias_correction=False,
            max_grad_norm=1.0,
        )
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        if args.from_pretrained:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                warmup=args.warmup_proportion,
                t_total=num_train_optimization_steps,

            )
        else:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                warmup=args.warmup_proportion,
                t_total=num_train_optimization_steps,
            )
        if args.continue_training:
            opt_state_dict_path = os.path.join(
                args.from_pretrained, "optimizer_state_{}.bin".format(int(args.start_epoch) - 1)
            )
            optimizer.load_state_dict(torch.load(opt_state_dict_path, map_location='cpu'))

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_dataset.num_dataset)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    model.eval()
    torch.set_grad_enabled(False)
    id2class = np.load("./dic/id2class1155.npy", allow_pickle=True).item()
    noun_size = len(id2class)
    print("Noun vocabulary size is {}".format(noun_size))
    prior_t = torch.zeros((noun_size), dtype=torch.float64).cuda()
    dic_t = torch.zeros((noun_size, 768), dtype=torch.float64).cuda()
    prior_v = torch.zeros((1601), dtype=torch.float64).cuda()
    dic_v = torch.zeros((1601, 2048), dtype=torch.float64).cuda()
    for step, batch in enumerate(train_dataset, 1):
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

        input_ids, image_feat, image_loc, segment_ids, input_mask, image_mask, image_target, num_boxes = batch
        embedding_output = model(
            input_ids,
            image_feat,
            image_loc,
            segment_ids,
            input_mask,
            image_mask,
        )

        l = input_ids.size(1)
        for i in range(input_ids.size(0)):
            for j in range(l):
                id = int(input_ids[i][j])
                cls = id2class.get(id)
                if cls is not None and (j != l-1 and int(input_ids[i][j+1]) not in special_ids or j == l-1):
                    prior_t[cls] += 1
                    dic_t[cls] += embedding_output[i][j].to(torch.float64)

        idx = torch.argmax(image_target, 2)
        for i in range(num_boxes.size(0)):
            for j in range(num_boxes[i]):
                index = idx[i][j]
                prior_v[index] += 1
                dic_v[index] += image_feat[i][j].to(torch.float64)

        if default_gpu and step % 20 == 0:
            print(step)


    np.save("./dic/prior_t_{}".format(dist.get_rank()), prior_t.cpu().numpy())
    np.save("./dic/dic_t_{}".format(dist.get_rank()), dic_t.cpu().numpy())
    np.save("./dic/prior_v_{}".format(dist.get_rank()), prior_v.cpu().numpy())
    np.save("./dic/dic_v_{}".format(dist.get_rank()), dic_v.cpu().numpy())

if __name__ == "__main__":
    main()
