from data_manager import Sentences
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
    model_1st = model.PartialModel(1)
    model_1st.fillModel(loadDataset())
    model.saveModel(model_1st)


def test():
    # now, only test for generating one word
    sentence = input("enter a sentence: ")
    model_1st = model.loadModel("pos1.pkl")
    w, p = model_1st.getWord(sentence)
    print(w, p)


if __name__ == "__main__":
    test()


