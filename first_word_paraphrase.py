import hmm
import torch
from PGM_basics import DictionaryFactor
from data_manager import *
import pickle

SAVE_PATH = "model/"

# Prepare dataset
train_set = Sentences("train")
train_loader = torch.utils.data.DataLoader(
    train_set, 
    batch_size = 1, 
    shuffle = False, 
)

def use_hmm():
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


def use_factor():
    dataset_length = len(train_set)
    n_vars = len(getWordDict()[0])
    # let input variable be index 1 - 17, and output be 21 - 37
    factor = DictionaryFactor([1, 21], [n_vars, n_vars])

    count = 0
    for sample in train_loader:
        # not that data and target is batched, but the batch size is 1
        # so use [0] to access this sample
        # [1] only mean get the first word. Note that [0] is <BOS>
        data = int(sample["input"][0][1])
        target = int(sample["target"][0][1])

        factor[(data, target)] = factor[(data, target)] + 1
        print("{:.5f}\t{}".format(count/dataset_length, factor[(1, 1)]), end = "\r")
        count += 1

    print()
    print("number of what:", factor[(1, 1)])
    print("number of I:", factor[(3, 3)])
    with open(SAVE_PATH + "pos1.pkl", "wb") as out_file:
        pickle.dump(factor, out_file)

if __name__ == "__main__":

    # what -> 1
    # i -> 3
    #use_factor()

    with open(SAVE_PATH + "pos1.pkl", "rb") as in_file:
        factor = pickle.load(in_file, encoding = "uft-8")
    print("number of what:", factor[(1, 1)])
    print("number of I:", factor[(3, 3)])