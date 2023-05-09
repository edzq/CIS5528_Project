import os
import sys
sys.path.append('../')
import logging
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from tools.data_preprocess import get_processor
from tools.dataset import NerDataset, ner_collate_fn
from tools.log_util import build_logger
from models.transformer_ner import TransformerNER
from train import Trainer
from layers.pretrained_model import PretrainedModel

import random


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
input_dir = os.path.join(parent_dir, 'dataset')
output_dir = os.path.join(parent_dir, 'output')
bert_dir = os.path.join(os.path.dirname(parent_dir), 'pretrained_models', 'save')


def main():
    
    sample_list = [0.1, 0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ner_processor = get_processor(dataset='NCBI')
    input = '/home/serc305/Research/NLP_Algorithm/NER/dataset/NCBI'
    train_examples = ner_processor.get_examples(os.path.join(input, 'train.jsonl'), data_type='train')
    idx = np.random.choice(np.arange(len(train_examples)), int(len(train_examples)*0.1), replace=False)
    train_examples_sample = train_examples[:int(len(train_examples)*0.1)]
    # print(train_examples[0])
    print(len(train_examples))
    print(len(train_examples_sample))

    # valid_examples = ner_processor.get_examples(os.path.join(args.input, 'dev.jsonl'), data_type='valid')
    # test_examples = ner_processor.get_examples(os.path.join(args.input, 'test.jsonl'), data_type='test')

    # label2id = {label: i for i, label in enumerate(ner_processor.get_labels(), 0)}

    # train_dataset = NerDataset(examples=train_examples, label2id=label2id, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    # valid_dataset = NerDataset(examples=valid_examples, label2id=label2id, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    # test_dataset = NerDataset(examples=test_examples, label2id=label2id, tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    # train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=ner_collate_fn)
    # valid_data_loader = DataLoader(valid_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=ner_collate_fn)
    # test_data_loader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=ner_collate_fn)

    # logger.info(f'Complete building dataset, train/valid/test: {len(train_examples)}/{len(valid_examples)}/{len(test_examples)}')

    # model = TransformerNER(enocder=encoder, rnn=args.rnn, crf=args.crf, hidden_dim=config.hidden_size, dropout=config.hidden_dropout_prob, tag_num=len(ner_processor.labels))

    # train_steps = (int(len(train_examples) / args.train_batch_size) + 1) * args.epochs

    # trainer = Trainer(cfg=args, model=model, label2id=label2id, train_steps=train_steps, basic_logger=logger)

    # trainer.train(args, train_data_loader, valid_data_loader)

    # trainer.evaluate(test_data_loader, type='test')


if __name__ == '__main__':

    main()














