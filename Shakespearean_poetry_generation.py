# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:20:02 2020

@author: Devdarshan
"""
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import Regularizer
import tensorflow.keras.utils as ku 
import numpy as np 




data = open('shakespear_poetry.txt').read()
corpus = data.lower().split("\n")


tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# convert text to sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

label = ku.to_categorical(label, num_classes=total_words)

# Model creation
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_len-1))  
model.add(Bidirectional(LSTM(150, return_sequences=True)))  
model.add(Dropout(0.2)) 
model.add(LSTM(100)) 
model.add(Dense(total_words/2, activation='relu'))  
model.add(Dense(total_words, activation='softmax')) 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])  #(# Pick a loss function and an optimizer)
print(model.summary())

history = model.fit(predictors, label, epochs=250, verbose=1)

# save model
model.save("mytrainedmodel.h5")

# poetry generation
poem=""
line = corpus[random.randint(0, len(corpus))] 
words = line.split() 
randomword = random.choice(words)
next_words = 5
total_lines=10
output_word=""
for i in range(total_lines):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output_word])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        randomword += " " + output_word
    output_word = randomword
    randomword=""
    poem += output_word + "\n"
    print(output_word)

# save poem
tfile = open("Poem.txt", "w")
tfile.write(poem)
tfile.close()
