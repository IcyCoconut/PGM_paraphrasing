from data_manager import Sentences, MAX_LENGTH
import torch
import model

# We can divide model into different parts (actually 15 parts)
# Let input words be X = {x1, x2, x3, ... x15}
# Let output words be Y = {y1, y2, y3, ... y15}
# Want to find factors f1(X,y1), f2(X,y2), f3(X,y3), ...

# For each factor fk, yk is a list of words rather than one word
# For example, f1(X, y1)
# y1 could be "what", "I", "where", "how", "he", "when", "why", "she" ...
# And we want to find P(y1 = "what" | X), P(y1 = "I" | X), P(y1 = "where" | X) ...
# And among all these probabilities we want to find the maximum one
# Let's say, P(y1 = "she" | X) is the greatest, then the first word should be "she"


def loadDataset():
    """ load and return dataloader """
    train_set = Sentences("train")
    loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size = 1, 
        shuffle = False, 
    )

    return loader


def buildModel():
    loader = loadDataset()
    for i in range(MAX_LENGTH):
        print("Building model X -> y{}".format(i + 1))
        part_model = model.PartialModel(i + 1)
        part_model.fillModel(loader)
        model.saveModel(part_model)


def test():
    # now, only test for generating one word
    sentence = "what can make math easy to learn" # input("enter a sentence: ")
    for i in range(MAX_LENGTH):
        part_model = model.loadModel("pos{}.pkl".format(i + 1))
        w, p = part_model.getWord(sentence)
        print(w[0], end = " ")
    print()


if __name__ == "__main__":
    buildModel()
    test()


