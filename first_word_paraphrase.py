import hmm
import torch
from data_manager import *

# Prepare dataset
train_set = Sentences("train")
train_loader = torch.utils.data.DataLoader(
    train_set, 
    batch_size = 1, 
    shuffle = False, 
)

# this will contain the indecies of
transitions = []
emitions = []
for sample in train_loader:
    # not that data and target is batched, but the batch size is 1, so use [0] to access this sample
    data = sample["input"][0]
    target = sample["target"][0]

    transitions.append(data[1])
    emitions.append(target[1])

sequence = [ (transitions, emitions) ]
model = hmm.train(sequence)

print(model.evaluate(["what"]))
print(model.decode(["what"]))