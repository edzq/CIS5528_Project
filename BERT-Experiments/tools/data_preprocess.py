import os
import pandas as pd
from typing import List, Set
from abc import ABC, abstractmethod


from .dataset import InputExample, InputFeatures


class NerProcessor(ABC):
    """Basic class for data converters for token-level sequence classification data sets."""

    @abstractmethod
    def get_examples(self, data_dir, data_type='train'):
        """Gets a collection of `InputExample`s for the train/valid/test set."""
        pass

    @abstractmethod
    def get_labels(self):
        """Gets the list of labels for this data set."""
        pass

    @abstractmethod
    def _create_examples(self, lines, data_type):
        """Creats a collection of `InputExample`s for the train/valid/test set."""
        pass

    @abstractmethod
    def _read_file(self, input_file):
        """Reads raw data from the original data file"""
        pass

class DatasetNERProcessor():
    def __init__(self):
        self.labels = set()

    def get_examples(self, data_path, data_type='train'):
        return self._create_examples(
            self._read_file(data_path), data_type
        )
    
    def get_labels(self):
        additional_labels = ['[CLS]', '[SEP]', 'X']
        print(self.labels)
        self.labels = list(self.labels)
        self.labels.extend(additional_labels)
        print(self.labels)
        return self.labels


    def _create_examples(self, data, data_type):
        examples = []
        for i, (sentence, label) in enumerate(data):
            uid = '%s-%s' % (data_type, i)
            text = ' '.join(sentence)
            label = label
            examples.append(InputExample(uid, text=text, label=label))
        return examples


    def _read_file(self, input_file):
        data = []

        #read the df['tokens'], df['labels']
        print(input_file)
        df = pd.read_json(path_or_buf=input_file, lines=True)
        tokens = df['tokens']
        bio_labels = df['labels']
        for token, label in zip(tokens, bio_labels):
            self.labels.update(label)
            data.append((token, label))
        return data

def get_processor(dataset):
    if dataset == 'DMDD':
        return DatasetNERProcessor()
    elif dataset == 'BC5CDR-chem':
        return DatasetNERProcessor()
    elif dataset == 'BC5CDR-disease':
        return DatasetNERProcessor()
    elif dataset == 'NCBI':
        return DatasetNERProcessor()
    elif dataset == 'BC2GM':
        return DatasetNERProcessor()