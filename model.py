from PGM_basics import DictionaryFactor, HmmFactor
from data_manager import *
import pickle
import torch
import sys
from time import time

# the variable index for x nodes and y nodes
# the graph is like a fully connected bipartite graph
X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
Y = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

# the path to save model
SAVE_PATH = "model/"


def saveModel(model, file_name):
    """ given a model, save it as a pkl file """
    with open(SAVE_PATH + file_name, "wb") as out_file:
        # wo do not want to save redundant data, so keys and vals are excluded
        pickle.dump(model, out_file)
    print("model save to", SAVE_PATH + file_name)


def loadModel(file_name):
    """ given a file anme, load and return the corresponding model from the given file """
    with open(SAVE_PATH + file_name, "rb") as in_file:
        model = pickle.load(in_file, encoding = "uft-8")
    print("{} loaded".format(file_name))
    return model



class AllToAllModel():
    """
    output(target) word: W = {w1, w2, w3, ... w15}
    input(observe) word: O = {o1, o2, o3, ... o15}

    255 factors: f(wi, oj), 1<=i<=15, 1<=j<=15
    14 transition factors: f(wi-1, wi), i = 2, 3, 4, ... 15

    w1 - w2 - w3 - ... - w8 - ... - w13 - w14 - w15
    |    |    |    ...   |    ...    |     |     |
    +----+----+--- ... --O--  ... ---+-----+-----+
    """
    def __init__(self):
        self.model_parts = []
        for i in range(15):
            self.model_parts.append(AllToOneModel(i + 1))


    def learnDataset(self, data_loader):
        """
        Learn the dataset, populate all factors that are store in all models
        """
        print("learning dataset")

        count = 0
        for sample in data_loader:
            input_sentence = sample["input"][0]
            target_sentence = sample["target"][0]

            prev_word = None
            for word_idx in range(2, 16):
                target_word = int(target_sentence[word_idx])
                self.model_parts[word_idx].populateFactors(input_sentence, target_word, prev_word)
                prev_word = target_word

            print("{}/127940".format(count), end = "\r")
            count += 1
        print("127940/127940")

        for i in range(15):
            self.model_parts[i].fixed()


    def assignModels(self, models):
        """
        If all AllToOneModel are trained and stored separately
        We can load these models separately and use this function to put them together
        """
        self.model_parts = models


    def getOutput(self, input_sentence):
        # TODO: use self.model_parts to get each single words and put then together as a sentence
        pass



# we actually need 15 this models to generate the full sentence, so I call this partial
class AllToOneModel():
    """
    output(target) word: wi
    input(observe) words: O = {o1, o2, o3, ... o15}

    15 factors: f(wi, ok), k = 1, 2, 3, 4, ... 15
    1 transition factor: f(wi-1, wi) only if i != 1
    
    wi-1 (if i != 1)------+
                          |
    +----+----+--- ... ---wi--- ... ----+----+----+
    |    |    |    ...    |     ...    |    |    |
    o1   o2   o3   ...    o8    ...   o13  o14  o15
    """
    def __init__(self, output_idx):
        self.factors = []
        for i in range(15):
            self.factors.append(HmmFactor())

        self.transition = HmmFactor() # this is f(wi-1, wi)
        self.output_idx = output_idx # this is i in wi

    
    def populateFactors(self, input_sentence, target_word, prev_word = None):
        """
        input sentence is o1 to o15
        target_word is wi
        prev_word is wi-1
        """

        for word_idx in range(1, 16):
            input_word = int(input_sentence[word_idx])
            self.factors[word_idx - 1][(input_word, target_word)] += 1

        self.transition[(prev_word, target_word)] += 1

    
    def fixed(self):
        """
        When all data is learnt, the entire dataset is studied
        We do not need to change all factors
        So call fixed() method on each factor, that makes observation and getWord faster
        and fixed() also normalizes these factors
        """
        for i in range(15):
            self.factors[i].fixed()
        self.transition.fixed()


    def getWord(self, input_sentence, prev_word = None):
        """
        Given the entire input sentence {o1, o2, o3, ..., o15}
        Generate and return a single output word wi
        """

        all_words = torch.zeros(0)
        all_probs = torch.zeros(0)

        for i in range(15):
            observe_word = input_sentence[i]

            words, probs = self.factors[i].observe(observe_word)

            # join factors
            all_words, idx = torch.unique(torch.cat((all_words, words)), return_inverse = True)
            concat_probs = torch.cat((all_probs, probs))
            new_probs = torch.zeros_like(words)
            for j in range(concat_probs.size(0)):
                new_probs[idx[j]] = concat_probs[j]
            all_probs = new_probs

        if prev_word != None:
            words, probs = self.transition.observe(prev_word)
            # join factors
            all_words, idx = torch.unique(torch.cat((all_words, words)), return_inverse = True)
            concat_probs = torch.cat((all_probs, probs))
            new_probs = torch.zeros_like(words)
            for j in range(concat_probs.size(0)):
                new_probs[idx[j]] = concat_probs[j]
            all_probs = new_probs

        # now all_words and all_probs contains all posible words with its probability
        try:
            chosen_idx = torch.argmax(all_probs)
            result = all_words[chosen_idx]
        except:
            result = input_sentence[self.output_idx]

        return result



class HiddenMarkovModel():
    """
    output(target) words: W = {w1, w2, w3, ... w15}
    input(observe) words: O = {o1, o2, o3, ... o15}

    14 transition factors: f(wk, wk+1), k = 1, 2, 3, ... 14
    15 emission factors: f(ok, wk), k = 1, 2, 3, ... 15

    w1 - w2 - w3 - w4 - ... - w15
    |    |    |    |    ...    |
    o1   o2   o3   o4   ...   o15

    """
    def __init__(self):
        # 15 emission factors phi(oi, wi)
        self.emiss_factors = []
        for i in range(15):
            self.emiss_factors.append(HmmFactor())

        # transition factors phi(wi, wi-1)
        self.trans_factors = []
        for i in range(14):
            self.trans_factors.append(HmmFactor())


    def learnDataset(self, data_loader):
        """
        Say we have 1,000,000 sentence pairs, each sentence has length 15 
        If factor is (o3, w3), then
            all_words_1 are 1,000,000 words at position 3 of every ground truth sentence
            all_words_2 are 1,000,000 words at position 3 of every input sentence
        """

        print("learning dataset")
        # we have 127940 sentences in total
        count = 0
        for sample in data_loader:
            input_sentence = sample["input"][0]
            target_sentence = sample["target"][0]

            # NOTE: target_word & input_word are actually indecies of words, instead of word strings
            # NOTE: the first word has index 1
            first_target = int(target_sentence[1])
            first_input = int(input_sentence[1])

            self.emiss_factors[0][(first_input, first_target)] += 1

            prev_target = first_target
            for word_idx in range(2, 16):
                # note that word_idx is 0 is always <BOS>
                target_word = int(target_sentence[word_idx])
                input_word = int(input_sentence[word_idx])

                self.emiss_factors[word_idx - 1][(input_word, target_word)] += 1
                self.trans_factors[word_idx - 2][(prev_target, target_word)] += 1
                prev_target = target_word

            print("{}/127940".format(count), end = "\r")
            count += 1
        print("127940/127940")

        # all data updated, no need to do any insertion
        for i in range(15):
            self.emiss_factors[i].fixed()
        for i in range(14):
            self.trans_factors[i].fixed()


    def getOutput(self, input_sentence):
        """
        input sentence is a list of ids, return an output sentence (ids)
        NOTE: the input sentence here should not have <BOS> sign at beginning
        """

        output = torch.ones(15) * EOS_ID

        for i in range(15):
            observe_word = input_sentence[i]

            words, probs = self.emiss_factors[i].observe(observe_word)

            if i > 1:
                trans_words, trans_probs = self.trans_factors[i-1].observe(observe_word)

                words, idx = torch.unique(torch.cat((words, trans_words)), return_inverse = True)
                concat_probs = torch.cat((probs, trans_probs))
                new_probs = torch.zeros_like(words)
                for j in range(concat_probs.size(0)):
                    new_probs[idx[j]] = concat_probs[j]
                probs = new_probs

            try:
                chosen_idx = torch.argmax(probs)
                output[i] = words[chosen_idx]
            except:
                output[i] = input_sentence[i]

        for i in range(15):
            if output[i] == UNK_ID:
                output[i] = input_sentence[i]

        return output
