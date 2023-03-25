# load dataset
from datasets import load_dataset
from nltk import pos_tag
from tqdm import tqdm, trange
from sklearn_crfsuite import CRF
import evaluate


def word2features(sent, i):
    '''
    Processing the each word in the sentences and extracting features
    Input:
        sent: a sentence, i.e. a list of tuples (word, postag, label)
        i: the index of the word
    Output:
        features: a dictionary of features
    '''
    word = str(sent[i][0])
    postag = str(sent[i][1])
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = str(sent[i-1][0])
        postag1 = str(sent[i-1][1])
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = str(sent[i+1][0])
        postag1 = str(sent[i+1][1])
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def get_dataset(dataset_name):
     # get train, validation, test data
    sets = ['train', 'validation', 'test']
    raw_datasets = load_dataset(dataset_name) # test ncbi_disease
    ner_feature = raw_datasets["train"].features["ner_tags"]
    label_names = ner_feature.feature.names

    # get all the sentences and labels and saving into one dic
    dataset = {
        'train': {
            'X':[],
            'y':[]
        },
        'validation':  {
            'X':[],
            'y':[]
        },
        'test':  {
            'X':[],
            'y':[]
        },
    }
    for set in sets:
        print(f"Loading {set} set ==>")
        for sent in tqdm(raw_datasets[f'{set}']):
            pos_sent = pos_tag(sent['tokens'])
            dataset[f'{set}']['X'].append(sent2features(pos_sent))
            ner_tags = [label_names[label] for label in sent['ner_tags']]
            dataset[f'{set}']['y'].append(ner_tags)
    return dataset, label_names

if __name__ == "__main__":
    # load dataset
    data_names = ['ncbi_disease', 'bc2gm_corpus', 'BC5CDR']
    dataset, label_names = get_dataset(dataset_name=data_names[0])

    # defining model
    crf = CRF(algorithm='lbfgs',
          c1=10,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)

    # training model
        # fit by sklearn style
    crf.fit(dataset['train']['X'], dataset['train']['y'])
    # test
    y_pred = crf.predict(dataset['test']['X'])
    # evalutate
    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=y_pred, references=dataset['test']['y'])
    print(results)

    



