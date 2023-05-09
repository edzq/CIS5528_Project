import os
import numpy as np
from tqdm import tqdm
from collections import Counter, OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab

from transformers import BertTokenizerFast

from .data_preprocess import *

class InputExample(object):
    """A single training/dev/test example."""

    def __init__(self, uid, text, label=None):
        """
        Args:
            uid(string): Unique id for the example.
            text(string): The untokenized text of the sequence.
            label(string, optional): The label of the example. This should be specified for train and dev examples.
        """
        self.uid = uid
        self.text = text
        self.label = label

class InputFeatures(object):
    """A single set of features of training/valid/test data."""
    def __init__(self, uid, tokens, input_ids, input_mask, segment_ids, label_ids, label_mask):
        self.uid = uid
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label_mask = label_mask

class NerDataset(Dataset):
    def __init__(self, examples, label2id=None, tokenizer=None, vocab=None, max_seq_length=512):
        super(NerDataset, self).__init__()
        self.features = convert_examples_to_features(examples, label2id, tokenizer, vocab, max_seq_length)

    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, index):
        feature = self.features[index]
        return feature.uid, feature.tokens, feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_ids, feature.label_mask

def convert_examples_to_features(examples: List[InputExample], label2id=None, tokenizer=None, vocab=None, max_seq_length=512):
    features = []
    for index, example in enumerate(examples):
        uid, text, label_list = example.uid, example.text, example.label
        word_list = text.split(' ')
        tokens = []
        labels = []
        # print(" ==== Debuging ===")
        # print(word_list)
        # print(label_list)
        if tokenizer is not None:
            for i, word in enumerate(word_list):
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label = label_list[i]
                for j in range(len(token)):
                    if j == 0:
                        labels.append(label)
                    else:
                        labels.append('X')
        else:
            #  不需要分词
            tokens = word_list
            labels = label_list
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0: max_seq_length - 2]
            labels = labels[0: max_seq_length - 2]

        new_tokens = []
        input_ids = []
        input_mask = []
        segment_ids = []
        label_ids = []
        label_mask = []
        if tokenizer is not None:
            new_tokens.append('[CLS]')
            segment_ids.append(0)
            label_ids.append(label2id['[CLS]'])
            label_mask.append(1)
            for i, token in enumerate(tokens):
                new_tokens.append(token)
                segment_ids.append(0)
                label_ids.append(label2id[labels[i]])
                if labels[i] == 'X':
                    label_mask.append(0)
                else:
                    label_mask.append(1)
            new_tokens.append("[SEP]")
            segment_ids.append(0)
            label_mask.append(1)
            label_ids.append(label2id["[SEP]"])
            input_ids = tokenizer.convert_tokens_to_ids(new_tokens)
            input_mask = [1] * len(input_ids)
        else:
            new_tokens = tokens
            input_ids = vocab(tokens)
            input_mask = [1] * len(input_ids)
            label_ids = [label2id[label] for label in labels]
            label_mask = [1] * len(label_ids)

        features.append(
            InputFeatures(uid=uid,
                          tokens=new_tokens,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids,
                          label_mask=label_mask
            )
        )

    return features

def ner_collate_fn(batch):
    uids, tokens, input_ids, input_mask, segment_ids, label_ids, label_mask = zip(*batch)
    max_length = max([len(input_id) for input_id in input_ids])
    # [ batch_size * max_seq_length ]
    input_ids = sequence_padding(input_ids, max_length)
    input_mask = sequence_padding(input_mask, max_length)
    segment_ids = sequence_padding(segment_ids, max_length)
    label_ids = sequence_padding(label_ids, max_length)
    label_mask = sequence_padding(label_mask, max_length)

    batch_input_ids = torch.tensor(input_ids, dtype=torch.long)
    batch_input_mask = torch.tensor(input_mask, dtype=torch.long)
    batch_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    batch_label_ids = torch.tensor(label_ids, dtype=torch.long)
    batch_label_mask = torch.tensor(label_mask, dtype=torch.long)

    return batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids, batch_label_mask


def sequence_padding(inputs, length=None, padding=0):
    """将序列padding到同一长度
    """
    if not inputs:
        return None

    outputs = np.array([
        np.concatenate([x, [padding] * (length - len(x))])
        if len(x) < length else x[:length] for x in inputs
    ])

    return outputs


def build_vocab(data):
    counter = Counter()
    for example in data:
        tokens = example.text.split(' ')
        counter.update(tokens)
    sorted_by_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq)
    _vocab = vocab(ordered_dict, specials=['<pad>', '<unk>'])
    _vocab.set_default_index(1)  # '<unk>' 和 oov token的索引
    return _vocab


if __name__ == '__main__':
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    root_dir = os.path.dirname(parent_dir)
    bert_cache_dir = os.path.join(root_dir, 'pretrained_models/save/bert-base-uncased')
    data_dir = os.path.join(parent_dir, 'dataset', 'DMDD')
    train_path = os.path.join(data_dir, 'train.jsonl')
    valid_path = os.path.join(data_dir, 'dev.jsonl')
    test_path = os.path.join(data_dir, 'test.jsonl')

    print(valid_path)

    ner_processor = DatasetNERProcessor()
    tokenizer = BertTokenizerFast.from_pretrained(bert_cache_dir)

    valid_examples = ner_processor.get_examples(valid_path, 'valid')
    #print(ner_processor.get_labels())

    label2id = {label: i for i, label in enumerate(ner_processor.get_labels(), 0)}
    print("labe2id dic: ", label2id)
    valid_dataset = NerDataset(valid_examples, label2id=label2id, tokenizer=tokenizer)
    print("Finish tokenization!")
    print(type(valid_dataset))
    uid, tokens, input_ids, input_mask, segment_ids, label_ids, label_mask = valid_dataset.__getitem__(0)
    print("uid: ", uid)
    print(tokens)
    print("input_ids: ", input_ids)
    print("input_masks: ",input_mask)
    print("segment_ids: ",segment_ids)
    print("labels_ids: ",label_ids)
    print("labels_mask: ",label_mask)

