from datasets import load_dataset
from tqdm import tqdm, trange
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
from keras.models import Model
from tensorflow.keras.layers import Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
import tensorflow as tf
from keras.metrics import Precision, Recall
import numpy as np
import evaluate
from keras.utils import to_categorical
# from tqdm.keras import TqdmCallback

def get_dataset(dataset_name):
     # get train, validation, test data
    sets = ['train', 'validation', 'test']
    raw_datasets = load_dataset(dataset_name) # test ncbi_disease
    ner_feature = raw_datasets["train"].features["ner_tags"]
    label2id = ner_feature.feature.names

    word_to_ix = {}
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
            sentence = sent['tokens']
            dataset[f'{set}']['X'].append(sentence)
            for word in sentence:
                if word not in word_to_ix:# word has not been assigned an index yet
                    word_to_ix[word] = len(word_to_ix) # Assign each word with a unique index
            #ner_tags = [label_names[label] for label in ]
            dataset[f'{set}']['y'].append(sent['ner_tags'])
    
    return dataset, label2id, word_to_ix

def post_process(dataset, word_to_ix, tag2id):
    sets = list(dataset.keys())
    max_len = 0
    print(" processing word id =======>")
    for set in tqdm(sets):
        dataset[f'{set}']['X'] = [[word_to_ix[w] for w in s] for s in dataset[f'{set}']['X']]
        dataset[f'{set}']['y'] = [[to_categorical(w, num_classes=3) for w in s] for s in dataset[f'{set}']['y']]
        sub_max = max([len(i) for i in dataset[f'{set}']['y']])
        if sub_max > max_len:
            max_len = sub_max
    
    # padding
    print(" processing sequence padding =======>")
    for set in tqdm(sets):
        dataset[f'{set}']['X'] = pad_sequences(maxlen=max_len, sequences= dataset[f'{set}']['X'], padding="post", value=word_to_ix["ENDPAD"])
        dataset[f'{set}']['y'] = pad_sequences(maxlen=max_len, sequences= dataset[f'{set}']['y'], padding="post", value=tag2id["O"])
    return dataset, max_len

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out


if __name__ == "__main__":
    data_names = ['ncbi_disease', 'bc2gm_corpus', 'BC5CDR']
    dataset, label2id, word_to_ix = get_dataset(dataset_name=data_names[0])
    tag2id = dict(zip(label2id, list(range(len(label2id)))))
    word_to_ix["ENDPAD"] = len(word_to_ix) # the corresponding padding
    words = word_to_ix.keys()
    ix_to_word = dict((v, k) for k, v in word_to_ix.items())
    dataset, max_len = post_process(dataset, word_to_ix, tag2id)

    n_words = len(word_to_ix.keys())
    n_tags = len(tag2id.keys())

    embedding_dim = 300 # v1 without pretrain, it was 50
    input_ = Input(shape=(max_len,))

    # define model
    model = Embedding(input_dim=n_words, output_dim=embedding_dim, 
                        input_length=max_len)(input_)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer
    model = Model(input_, out)

    # filepath = './checkpoints/'
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(dataset['train']['X'], dataset['train']['y'], batch_size=32, epochs=5, 
                        verbose=0, validation_data=(dataset['validation']['X'], dataset['validation']['y']))
    
    test_pred = model.predict(np.array(dataset['test']['X']), verbose=1)
    idx2tag = {i: w for w, i in tag2id.items()}    
    pred_labels = pred2label(test_pred)
    test_labels = pred2label(dataset['test']['y'])
    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=pred_labels, references=test_labels)
    print(results)