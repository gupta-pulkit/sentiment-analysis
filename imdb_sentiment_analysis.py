import numpy as np
import tensorflow as tf
print(tf.__version__)

import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []
for s, l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())
    
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)
print(training_labels_final.shape, testing_labels_final.shape)

vocab_size = 1000
embedding_dim = 16
max_length = 30
trunc_type = 'post'
oov_tok = "<oov>"

import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen = max_length, truncating = trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen = max_length, truncating = trunc_type)

def model_2(vocab_size, embedding_dim, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim,input_length = max_length))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    return model

model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(20)),
    tensorflow.keras.layers.Dense(10, activation = 'relu'),
    tensorflow.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

num_epochs = 20
history = model.fit(padded, training_labels_final, epochs = num_epochs, validation_data = (testing_padded, testing_labels_final))
