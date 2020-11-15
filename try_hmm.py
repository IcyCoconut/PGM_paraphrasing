import hmm

# this HMM model does not work for this project
# reason 1: it take too long to train, the algorithm is not efficient
# reason 2: sequence paraphrasing task does not really have a clear state transformation and emition


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