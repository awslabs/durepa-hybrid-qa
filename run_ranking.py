import glob
import os
import pandas as pd
import argparse
import numpy as np
import logging
from tqdm import tqdm, trange
import glob

from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from scipy import stats

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
# from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers

# transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from utils_ranking import generate_dataloader
import jsonlines


logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "AutoModelForSequenceClassification": (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer),
    "AutoModel": (AutoConfig, AutoModel, AutoTokenizer),
}


class Reranker(pl.LightningModule):

    def __init__(self, model, tokenizer, args):
        super(Reranker, self).__init__()

        self.hparams = args

        # model will be a BertForSequenceClassification model
        self.model = model
        self.tokenizer = tokenizer
        # self.train_dl = train_dl
        # self.val_dl = val_dl
        # self.test_dl = test_dl

        # calculate total training steps
        assert args.num_train_steps > 0
        self.t_total = args.num_train_steps

        if args.warmup_steps < 1.0:
            self.hparams.warmup_steps = int(args.warmup_steps * self.t_total)
        else:
            self.hparams.warmup_steps = int(args.warmup_steps)
            assert self.hparams.warmup_steps < self.t_total

        logger.info(f'Number of warmup steps: {self.hparams.warmup_steps}, total number of training steps: {self.t_total}')

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        return outputs

    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch

        # reshape
        input_ids = input_ids.reshape(-1, self.hparams.max_length) # [b, max_length*num_cand] --> [b*num_cand, max_length]
        attention_mask = attention_mask.reshape(-1, self.hparams.max_length)
        token_type_ids = token_type_ids.reshape(-1, self.hparams.max_length)

        # fwd
        outputs = self.forward(input_ids, attention_mask, token_type_ids)

        # loss
        logits = outputs[0] # (b*num_cand, 1)
        logits = logits.reshape(-1, self.hparams.num_cand) # (b, num_cand)
        # probability = torch.full((logits.size(1),), 0.8).type_as(logits.float()) # (num_cand,)
        # weights = torch.bernoulli(probability)
        # loss = F.cross_entropy(logits, label, weight=weights) # label: (b,)
        loss = F.cross_entropy(logits, label)

        return {'loss': loss}

    def training_step_end(self, outputs):
        loss = outputs['loss'].mean()
        # logs
        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch

        # reshape
        input_ids = input_ids.reshape(-1, self.hparams.max_length) # [b, max_length*num_cand] --> [b*num_cand, max_length]
        attention_mask = attention_mask.reshape(-1, self.hparams.max_length)
        token_type_ids = token_type_ids.reshape(-1, self.hparams.max_length)

        # fwd
        outputs = self.forward(input_ids, attention_mask, token_type_ids)

        # loss
        logits = outputs[0] # (b*num_cand, 1)
        logits = logits.reshape(-1, self.hparams.num_cand) # (b, num_cand)
        loss = F.cross_entropy(logits, label) # label: (b,)

        # prediction
        _, y_hat = torch.max(logits, dim=1) # (b,)

        return {'val_loss': loss, 'y_hat': y_hat, 'label': label}

    def validation_step_end(self, outputs):
        loss  = outputs['val_loss'].mean()
        y_hat = outputs['y_hat'] # (b,)
        label = outputs['label'] # (b,)

        val_performance = accuracy_score(label.cpu(), y_hat.cpu()) # recall@1 / accuracy
        val_performance = torch.tensor(val_performance)

        return {'val_loss': loss, 'val_performance': val_performance}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_performance = torch.stack([x['val_performance'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_performance': avg_val_performance}
        logger.info(f"Validation performance: {avg_val_performance}")

        return {'avg_val_loss': avg_loss, 'avg_val_performance': avg_val_performance, 'progress_bar': tensorboard_logs}

    # def test_step(self, batch, batch_nb):
    #     # batch
    #     input_ids, attention_mask, token_type_ids, label = batch
    #
    #     # fwd
    #     outputs = self.forward(input_ids, attention_mask, token_type_ids)
    #
    #     # loss
    #     logits = outputs[0] # (b, 1)
    #
    #     return {'logits': logits}
    #
    # def test_epoch_end(self, outputs):
    #     logits = torch.cat([x['logits'] for x in outputs], dim=0).cpu()
    #     scores = F.sigmoid(logits).squeeze(1).numpy().tolist() # (num_test, )
    #
    #     lines = []
    #     count = 0
    #     with jsonlines.open(os.path.join(self.hparams.data_dir, "test.es_retrieved.jsonl")) as f:
    #         for line in f.iter():
    #             for cand in line['passages']:
    #                 cand['rank_score'] = scores[count]
    #                 count += 1
    #                 if count % 10000 == 0:
    #                     logger.info(f"Finished {count} lines")
    #             lines.append(line)
    #
    #     with jsonlines.open(os.path.join(self.hparams.data_dir, "test.es_retrieved.scores.jsonl"), 'w') as writer:
    #         writer.write_all(lines)
    #
    #     return {'avg_test_loss': -1}

    def configure_optimizers(self):

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = {
             'scheduler': None,
             'monitor': 'val_loss', # Default: val_loss
             'interval': 'step', # step or epoch
             'frequency': 1
        }

        if self.hparams.lr_schedule == 'linear':
            scheduler['scheduler'] = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.t_total
            )
        elif self.hparams.lr_schedule == 'cosine':
            scheduler['scheduler'] = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.t_total
            )
        elif self.hparams.lr_schedule == 'cosine_hard':
            scheduler['scheduler'] = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.t_total, num_cycles=2.0
            )
        else:
            return optimizer

        return [optimizer], [scheduler]


    # @pl.data_loader
    # def train_dataloader(self):
    #     return self.train_dl
    #
    # @pl.data_loader
    # def val_dataloader(self):
    #     return self.val_dl

    # @pl.data_loader
    # def test_dataloader(self):
    #     return self.test_dl


class RerankerInference(pl.LightningModule):

    def __init__(self, model, tokenizer, args):
        super(RerankerInference, self).__init__()

        self.hparams = args

        # model will be a BertForSequenceClassification model
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        return outputs

    def test_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch

        # fwd
        outputs = self.forward(input_ids, attention_mask, token_type_ids)

        # loss
        logits = outputs[0] # (b, 1)

        return {'logits': logits}

    def test_epoch_end(self, outputs):
        logits = torch.cat([x['logits'] for x in outputs], dim=0).cpu()
        scores = F.sigmoid(logits).squeeze(1).numpy().tolist() # (num_test, )

        lines = []
        count = 0
        with jsonlines.open(os.path.join(self.hparams.data_dir, f"{self.hparams.question_type}.dev.es_retrieved.processed.jsonl")) as f:
            for line in f.iter():
                line['passages'] = line['passages'][:100]
                line['tables'] = line['tables'][:100]

                for cand in line['passages']:
                    cand['rank_score'] = scores[count]
                    count += 1
                    if count % 10000 == 0:
                        logger.info(f"Finished {count} lines")
                for cand in line['tables']:
                    cand['rank_score'] = scores[count]
                    count += 1
                    if count % 10000 == 0:
                        logger.info(f"Finished {count} lines")
                lines.append(line)

        score_dir = os.path.join(self.hparams.data_dir, "scores")
        if not os.path.exists(score_dir):
            os.makedirs(score_dir)
        with jsonlines.open(os.path.join(score_dir, f"{self.hparams.question_type}.dev.es_retrieved.processed.scores.jsonl"), 'w') as writer:
            writer.write_all(lines)

        return {'avg_test_loss': -1}

    # @pl.data_loader
    # def test_dataloader(self):
    #     return self.test_dl


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', nargs='*')
    parser.add_argument('--pretrained_model', default='bert-base-uncased', type=str)
    parser.add_argument('--overwrite', default=None, nargs='*')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--data_dir', default='/home/ec2-user/efs/ott-qa/', type=str)
    parser.add_argument('--num_cand', default=64, type=int)
    parser.add_argument('--question_type', default='ott-qa', choices=['opensquad', 'wikisql_denotation', 'NQ-open', 'ott-qa'])

    parser.add_argument('--task_type', default='AutoModelForSequenceClassification', type=str)
    parser.add_argument('--checkpoint_dir', default='/home/ec2-user/efs/ck/ott-qa/', type=str)
    parser.add_argument('--cache_dir', default='/home/ec2-user/efs/cache/', type=str)
    parser.add_argument('--tensorboard_dir', default='/home/ec2-user/efs/wandb/', type=str)
    parser.add_argument('--load_model_checkpoint', default=None, type=str, help="The checkpoint file upon which you want to continue training on.")
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--test_batch_size', default=1024, type=int)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--lr_schedule', default='linear', type=str, choices=['linear', 'cosine', 'cosine_hard', 'constant'])
    parser.add_argument('--warmup_steps', default=0.1, type=float, help="if < 1, it means fraction; otherwise, means number of steps")
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int)
    parser.add_argument('--num_train_epochs', default=3, type=int)
    parser.add_argument('--num_train_steps', default=10000, type=int)
    parser.add_argument('--max_length', default=150, type=int)

    # add all the available options to the trainer
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.question_type)
    logger.info(f"Checkpoint directory: {args.checkpoint_dir}")

    if args.gpu == None:
        logger.info("not using GPU")
        args.gpu = 0
    else:
        try:
            args.gpu = [int(x) for x in args.gpu]
            logger.info(f"using gpu {args.gpu}")
        except:
            ValueError("only support numerical values")

    # read pretrained model and tokenizer using config
    logger.info("loading pretrained model and tokenizer")
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.task_type]
    config = config_class.from_pretrained(args.pretrained_model, cache_dir=args.cache_dir)
    config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(args.pretrained_model, use_fast=True, cache_dir=args.cache_dir)
    model = model_class.from_pretrained(
                args.pretrained_model,
                from_tf=False,
                config=config,
                cache_dir=args.cache_dir)

    # add special tokens
    additional_special_tokens_dict = {'additional_special_tokens': ['[title]']}
    tokenizer.add_special_tokens(additional_special_tokens_dict) # add classification tokens
    model.resize_token_embeddings(len(tokenizer))

    if args.overwrite is None:
        args.overwrite = []

    # checkpoint
    checkpoint_dir = os.path.join(args.checkpoint_dir, f'{args.pretrained_model}/')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_callback = ModelCheckpoint(monitor='avg_val_performance', filepath=checkpoint_dir+'{epoch}-{val_loss:.4f}-{avg_val_performance:.4f}', mode='max')

    # training and testing
    if args.do_train:
        # initialized dataloaders
        train_dataloader = generate_dataloader(
            args.data_dir,
            tokenizer,
            args.max_length,
            'train',
            args.num_cand,
            'train' in args.overwrite,
            args.batch_size,
            args.question_type,
        )
        val_dataloader = generate_dataloader(
            args.data_dir,
            tokenizer,
            args.max_length,
            'dev',
            args.num_cand,
            'dev' in args.overwrite,
            args.batch_size,
            args.question_type,
        )
        # test_dataloader = generate_dataloader(
        #     args.data_dir,
        #     tokenizer,
        #     args.max_length,
        #     'test',
        #     args.num_cand,
        #     'test' in args.overwrite,
        #     args.batch_size,
        #     args.question_type,
        # )
        # test_dataloader = None

        if args.num_train_steps <= 0:
            args.num_train_steps = len(train_dataloader) * args.num_train_epochs

        bert_ranker = Reranker(model, tokenizer, args)
        if args.load_model_checkpoint != None:
            logger.info(f"Loading the checkpoint {args.load_model_checkpoint} and continue training")
            model_checkpoint = torch.load(args.load_model_checkpoint, map_location=lambda storage, loc: storage)
            model_dict = model_checkpoint['state_dict']
            bert_ranker.load_state_dict(model_dict)

        tb_logger = loggers.WandbLogger(save_dir=args.tensorboard_dir, project='hybridQA-ott-qa')
        trainer = pl.Trainer(logger=tb_logger,
                             checkpoint_callback=checkpoint_callback,
                             gpus=args.gpu,
                             distributed_backend='dp',
                             val_check_interval=0.25, # check every certain % of an epoch
                             # min_epochs=args.num_train_epochs,
                             max_epochs=args.num_train_epochs,
                             max_steps=args.num_train_steps,
                             accumulate_grad_batches=args.gradient_accumulation_steps,
                             gradient_clip_val=1.0,
                             precision=args.precision)        # train
        trainer.fit(bert_ranker, train_dataloader, val_dataloader)
        # trainer.test(bert_ranker)

    if args.do_test:
        torch.cuda.empty_cache()

        # initialized dataloaders
        test_dataloader = generate_dataloader(
            args.data_dir,
            tokenizer,
            args.max_length,
            'test',
            args.num_cand,
            'test' in args.overwrite,
            args.test_batch_size,
            args.question_type,
        )

        if args.load_model_checkpoint:
            best_checkpoint_file = args.load_model_checkpoint
        else:
            # find best checkpoint
            best_val_performance = -100.
            best_val_loss = 100.
            for checkpoint_file in glob.glob(checkpoint_dir+"*avg_val_performance*.ckpt"):
                val_performance = float(checkpoint_file.split('=')[-1].replace('.ckpt',''))
                val_loss = float(checkpoint_file.split('=')[-2].split('-')[0])
                if val_performance > best_val_performance:
                    best_val_performance = val_performance
                    best_val_loss = val_loss
                    best_checkpoint_file = checkpoint_file
        logger.info(f"Loading the checkpoint: {best_checkpoint_file}")

        # load model
        bert_ranker = RerankerInference(model, tokenizer, args)
        best_checkpoint = torch.load(best_checkpoint_file, map_location=lambda storage, loc: storage)
        bert_ranker.load_state_dict(best_checkpoint['state_dict'])

        # test using Trainer test function
        trainer = pl.Trainer(gpus=args.gpu, distributed_backend='dp', benchmark=True)
        trainer.test(bert_ranker, test_dataloader)


if __name__ == "__main__":
    main()
