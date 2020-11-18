from PGM_basics import DictionaryFactor
from data_manager import *
import pickle
import torch

# the variable index for x nodes and y nodes
# the graph is like a fully connected bipartite graph
X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
Y = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

# the path to save model
SAVE_PATH = "model/"


def saveModel(model):
    """ given a model, save it as a pkl file """
    with open(SAVE_PATH + "pos{}.pkl".format(model.pos), "wb") as out_file:
        pickle.dump(model, out_file)
    print("model save to", SAVE_PATH + "pos{}.pkl".format(model.pos))


def loadModel(file_name):
    """ given a file anme, load and return the corresponding model from the given file """
    with open(SAVE_PATH + file_name, "rb") as in_file:
        model = pickle.load(in_file, encoding = "uft-8")
    return model


# we actually need 15 this models to generate the full sentence, so I call this partial
class PartialModel():
    def __init__(self, pos: int):
        """
        Example: If pos = 2,  we want to get the 2nd word of paraphrase sentence, which is Y[1]
        If we want 15 factors:
            f(X[0], Y[1]): given the word at X[0], probabilities of all words at Y[1]
            f(X[1], Y[1]): given the word at X[1], probabilities of all words at Y[1]
            f(X[2], Y[1]): given the word at X[2], probabilities of all words at Y[1]
            ... X[3] to X[13]
            f(X[14], Y[1]): given the word at X[14], probabilities of all words at Y[1]
        """
        n_vars = len(getWordDict()[0])
        self.pos = pos
        self.factor_list = []
        for factor_idx in range(MAX_LENGTH):
            self.factor_list.append(
                DictionaryFactor([X[factor_idx], Y[pos - 1]], [n_vars, n_vars])
            )

        self.c = 0.7


    def fillModel(self, data_loader):
        """
        Read the dataset and fill the model, fill the factors
        For example, if the input words are "I want to sleep" and the target words are "I will nap"
        self.pos = 2, the target word is "will"
        So, P("will" | "I"), P("will", | "want"), P("will", "to"), P("will", "sleep") will increase
         P("will", "<EOS>") will not increase
        """
        print("Filling model with data\ndata filled:")
        count = 0
        for sample in data_loader:
            # not that data and target is batched, but the batch size is 1
            # so use [0] to access this sample
            # [1] only mean get the first word. Note that [0] is <BOS>
            input_sentence = sample["input"][0]
            target_sentence = sample["target"][0]

            # NOTE: target_word & input_word are actually indecies of words, instead of word strings
            target_word = int(target_sentence[self.pos])

            for word_idx in range(MAX_LENGTH):
                # note that word_idx is 0 is always <BOS>
                input_word = int(input_sentence[word_idx + 1])
                if input_word == EOS_ID:
                    break
                self.factor_list[word_idx][(input_word, target_word)] += 1

            print("{}\t".format(count), end = "\r")
            count += 1
        print("\ndone")


    def getWord(self, input_sentence):
        """
        input_sentence is a sentence! It should be a string or list of words, not indicies of words
        given the input sentence, the function choose and return a word, with its occurances
        """

        ids = wordsToIds(input_sentence)
        # word_choices in format: {word_index : probability}
        word_choices = dict()
        for i in range(MAX_LENGTH):
            # observe ith word is ids[i]
            self.factor_list[i].observe(X[i], ids[i])
            # distance is how far away this word is to the target word
            # further distance may lead to less connection, so less weight
            distance = abs(self.pos - (i+1))
            # iterate all possible words, if the word not in word choices then add it, if exists then update the value
            for k in self.factor_list[i].dictionary:
                # k is the index for a word
                if word_choices.get(k) == None:
                    word_choices[k] = distance * self.c * self.factor_list[i][k]
                else:
                    word_choices[k] += distance * self.c * self.factor_list[i][k]
        
        # new word_choices should contain all possible word choices and their probability
        # for this position, we want the maximum one for now
        max_k = list(word_choices.keys())[0]
        for k in word_choices:
            if word_choices[k] > word_choices[max_k]:
                max_k = k
        
        return idsToWords(list(max_k)), word_choices[max_k]


