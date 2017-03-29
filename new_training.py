import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import time

filename = "frost_train_data.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

#########################################################################################################
##############################################Data Preparation###########################################
#########################################################################################################

#Creating a list of all characters present in the file
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

#Number of characters in the dataset
n_chars = len(raw_text)
#Vocabulary size of the dataset
n_vocab = len(chars)

#Length in which the senetences will be broken
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i: i + seq_length]
    seq_out = raw_text[i + seq_length]
    #Converting characters to integers using char_to_int dictionary
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print n_patterns

#########################################################################################################
##############################################Data Pre-processing########################################
#########################################################################################################

#Input sequence for LSTM network : [samples, time steps, features]
#rescale integers from 0-1 since LSTM uses sigmoid to squash the activation values in [0,1]
#convert the output patterns into a one h0t encoding


# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
#normalize
X = X / float(n_vocab)
#one-hot encode the output variable
y = np_utils.to_categorical(dataY)
print y.shape
#defining the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

'''
Because of the slowness and because of our optimization requirements, we will use model checkpointing to
record all of the network weights to file each time an improvement in loss is observed at the end of the epoch.
We will use the best set of weights (lowest loss) to instantiate our generative model in the next section.
'''
filepath = "weights-improvement={epoch:02d}-{loss:4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print "Training started : "
start_time = time.time()
model.fit(X, y, nb_epoch=800, batch_size=100, callbacks=callbacks_list, verbose=1)
end_time = time.time()
print "Training complete."
print "Time taken : ", (end_time - start_time)
