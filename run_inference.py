import glob
import os
import argparse
import numpy as np
import logging
import glob
import jsonlines
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import recall_score
from typing import Iterable, Optional, Tuple
import torch
import torch.nn.functional as F
# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers, seed_everything
import pdb

from src.utils_fusion_in_decoder import generate_dataloader
from src.run_fusion_in_decoder import T5

# transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='/home/ec2-user/efs/FID')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_checkpoint', type=str, required=True)
    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--num_return_sequences', type=int, default=3)
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f'{args.config}')
    cfg.data.data_dir = args.data_dir
    if args.output_dir is None:
        args.output_dir = os.path.join('/home/ec2-user/efs/hybrid_qa_inference/', os.path.basename(args.config).replace('.yaml', ''))
        print(args.output_dir)

    if args.test_batch_size is not None:
        cfg.optim.test_batch_size = args.test_batch_size
    if args.overwrite_cache:
        cfg.data.overwrite_cache = 'true'

    print(args.output_dir)
    cfg.data.output_dir = args.output_dir
    cfg.model.model_checkpoint = args.model_checkpoint
    cfg.data.num_beams = args.num_beams
    cfg.data.num_return_sequences = args.num_return_sequences

    os.makedirs(cfg.data.output_dir, exist_ok=True)

    # set seed
    seed_everything(cfg.optim.seed)

    tokenizer = T5Tokenizer.from_pretrained(
        cfg.model.tokenizer_name if cfg.model.tokenizer_name else cfg.model.model_name,
        cache_dir=cfg.model.cache_dir,
        use_fast=cfg.model.use_fast,
    )

    model_t5 = T5(cfg, tokenizer)

    logger.info("Evaluation starts")

    test_dataloader = generate_dataloader(
        data_dir = cfg.data.data_dir,
        tokenizer = tokenizer,
        max_source_length = cfg.data.max_source_length,
        max_target_length = cfg.data.max_target_length,
        overwrite_cache = cfg.data.overwrite_cache,
        mode = "test",
        batch_size = cfg.optim.test_batch_size,
        question_type = cfg.data.question_type,
        passage_type = cfg.data.passage_type,
        enable_sql_supervision = cfg.data.enable_sql_supervision,
        cand_for_each_source = cfg.data.cand_for_each_source,
    )

    torch.cuda.empty_cache()

    best_checkpoint_file = cfg.model.model_checkpoint

    # load model
    best_checkpoint = torch.load(best_checkpoint_file, map_location=lambda storage, loc: storage)
    model_t5.load_state_dict(best_checkpoint['state_dict'])

    # test using Trainer test function
    # cfg.trainer.precision = 32
    trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True))
    trainer.test(model_t5, test_dataloader)


if __name__ == "__main__":
    main()
