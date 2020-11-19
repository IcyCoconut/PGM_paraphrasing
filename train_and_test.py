from data_manager import Sentences, MAX_LENGTH, idsToWords, wordsToIds
import torch
import model
from datetime import datetime

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


def loadDataset(bs = 1):
    """ load and return dataloader """
    train_set = Sentences("train")
    loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size = bs, 
        shuffle = False, 
    )

    return loader


def writeLog(message: str):
    """ Given a message, this fucntion write the message to log.txt together with date and time """
    print(str(datetime.now()) + "\t" + message)
    log_file = open("log.txt", "a")
    log_file.write(str(datetime.now()) + "\t" + message + "\n")
    log_file.close()


def buildModel():
    """ initialize a model """
    loader = loadDataset()
    for i in range(MAX_LENGTH):
        print("Building model X -> y{}".format(i + 1))
        part_model = model.PartialModel(i + 1)
        part_model.fillModel(loader)
        model.saveModel(part_model)


def learnModel():
    """ update probabilities in the model """
    loader = loadDataset(bs = 10)
    for i in range(3, 5):
        writeLog("Learning model X -> y{}".format(i + 1))

        part_model = model.loadModel("pos{}.pkl".format(i + 1))
        part_model.learn(loader)
        model.saveModel(part_model)

        writeLog("Model X -> y{} saved to pos{}.pkl".format(i + 1, i + 1))


def test():
    sentence = "what can make physics easy to learn" # input("enter a sentence: ")
    sentence = wordsToIds("what can make maths easy to learn")[1:-1]
    print(sentence)
    prev_word = None
    for i in range(MAX_LENGTH):
        part_model = model.loadModel("pos{}.pkl".format(i + 1))
        w = part_model.getWord(sentence, prev_word)
        #w = part_model.getWordFast(sentence.to("cuda"), prev_word)
        prev_word = w
        print(idsToWords([w]))


if __name__ == "__main__":
    # buildModel()
    # learnModel()
    test()


