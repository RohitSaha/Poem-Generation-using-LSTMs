#Code with data of all poets and functionality

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys


def sample(preds, temperature):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

'''
Building a single LSTM depending on code received from client.
Code Author
1    Shakespeare
2    Robert Frost
3    T.S Elliot
'''

def accept_request(tag, poet_id):
    if poet_id == 1:
        text = open('train_data/Shakespeare_poem_train_set.txt').read().lower()
    elif poet_id == 2:
        text = open('train_data/frost_train_data.txt').read().lower()
    elif poet_id == 3:
        text = open('train_data/t.s_eliot_train_data.txt').read().lower()
    print('corpus length:', len(text))
    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print("Building model")
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    if poet_id == 1:
        model.load_weights('weights/shakespeare_LSTM_weights_3.h5')
    elif poet_id == 2:
        model.load_weights('weights/frost_LSTM_weights_3.h5')
    elif poet_id == 3:
        model.load_weights('weights/ts_eliot_LSTM_weights_3.h5')
    print("Model complete")



    # Since maxlen = 40, we have to feed words/sentences which are of length = 40
    sentence = tag
    sentence = str(sentence).lower()
    if len(sentence) < 40:
        get_diff = 40 - len(sentence)
        while get_diff > 0:
            sentence = ' ' + sentence
            get_diff -= 1


    generated = ''
    generated += sentence
    print('Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(300):
        x = np.zeros((1, len(sentence), len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, 0.1)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        #sys.stdout.write(next_char)
        #sys.stdout.flush()
    #print()
    print(generated)
    return generated