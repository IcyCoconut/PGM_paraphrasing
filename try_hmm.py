import hmm
from train_and_test import loadDataset
from data_manager import *
import torch
import pickle

# this HMM model does not work for this project
# reason 1: it take too long to train, the algorithm is not efficient
# reason 2: sequence paraphrasing task does not really have a clear state transformation and emition
"""

# for word at position 1
states = ('rainy', 'sunny')
symbols = ('walk', 'shop', 'clean')

trans_prob = {
    'rainy': { 'rainy' : 0.7, 'sunny' : 0.3 },
    'sunny': { 'rainy' : 0.4, 'sunny' : 0.6 }
}

emit_prob = {
    'rainy': { 'walk' : 0.1, 'shop' : 0.4, 'clean' : 0.5 },
    'sunny': { 'walk' : 0.6, 'shop' : 0.3, 'clean' : 0.1 }
}

model = hmm.Model(states, symbols, trans_prob = trans_prob, emit_prob = emit_prob)

sequences = [ (["rainy", "rainy"], ["walk", "shop"]), (["rainy", "rainy", "rainy"], ["walk", "walk", "shop"]) ]

model = hmm.train(sequences)

sequence = ["walk"]

print(model.evaluate(sequence))
print(model.decode(sequence))
"""

trans_probs = []
emit_probs = []
dictionary = getWordDict()
loader = loadDataset()
count = 0

# load all data
for sample in loader:
    input_sentence = sample["input"][0]
    target_sentence = sample["target"][0]

    for idx in input_sentence:
        trans_probs.append(dictionary[1][int(idx)])

    for idx in target_sentence:
        emit_probs.append(dictionary[1][int(idx)])

    print("{}/127940".format(count), end = "\r")
    count += 1
print()

# train hmm model
model = hmm.train([(trans_probs, emit_probs),])

# save model
with open("model/hmm_model.pkl", "wb") as out_file:
    pickle.dump(model, out_file)

