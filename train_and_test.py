from data_manager import Sentences, MAX_LENGTH, idsToWords, wordsToIds
import torch
import model
import pickle
from datetime import datetime
import evaluate as eva

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


def loadDataset(bs = 1, set_name = "train"):
    """ load and return dataloader """
    train_set = Sentences(set_name)
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


def all2All_train_save():
    # TODO: train and save an AllToAllModel
    writeLog("start train AllToAllModel")
    ata_model = model.AllToAllModel()
    loader = loadDataset()
    ata_model.learnDataset(loader)

    ata_model.saveToParts()
    writeLog("AllToAllModel saved")

    sentence = wordsToIds("how do i know someone blocked me on facebook")[1:-1]
    print(idsToWords(sentence))
    ata_model.setWeights(0.8)
    output = ata_model.getOutput(sentence)
    print(idsToWords(output))


def test_all2all():
    m = model.AllToAllModel()
    m.loadParts()
    sentence = wordsToIds("what is your name")[1:-1]
    print(idsToWords(sentence))
    m.setWeights(0.8)
    output = m.getOutput(sentence)
    print(idsToWords(output))

    loader = loadDataset(set_name = "test")

    with open("data/testdata/result.txt", "a") as result_file:
        c = 0
        for sample in loader:
            if c > 3767:
                input_sentence = sample["input"][0][1:-1]
                #target_sentence = sample["target"][0]

                output = idsToWords(m.getOutput(input_sentence))
                
                try:
                    last_index = output.index("<EOS>")
                    line = " ".join(output[:last_index]) + "\n"
                except:
                    line = " ".join(output) + "\n"
                result_file.write(line)

            print(c, end = "\r")
            c += 1


def hmm_train_save():
    # define a hmm
    m = model.HiddenMarkovModel()
    # load dataset
    loader = loadDataset()
    # learn model
    m.learnDataset(loader)
    # just a very simple test
    sentence = wordsToIds("what is your name")[1:-1]
    print(idsToWords(sentence))
    output = m.getOutput(sentence)
    print(idsToWords(output))

    # save the model
    model.saveModel(m, "hmm.pkl")


def test_hmm():
    m = model.loadModel("hmm.pkl")
    sentence = wordsToIds("what is your name")[1:-1]
    print(idsToWords(sentence))
    output = m.getOutput(sentence)
    print(idsToWords(output))

    loader = loadDataset(set_name = "test")

    with open("data/testdata/result.txt", "a") as result_file:
        c = 0
        for sample in loader:
            if c > 3767:
                input_sentence = sample["input"][0][1:-1]
                #target_sentence = sample["target"][0]

                output = idsToWords(m.getOutput(input_sentence))
                
                try:
                    last_index = output.index("<EOS>")
                    line = " ".join(output[:last_index]) + "\n"
                except:
                    line = " ".join(output) + "\n"
                result_file.write(line)

            print(c, end = "\r")
            c += 1

if __name__ == "__main__":
    # buildModel()
    # learnModel()
    # test()
    # hmm_train_save()
    # test_hmm()
    # all2All_train_save()
    test_all2all()


