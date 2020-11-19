from PGM_basics import DictionaryFactor
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


def saveModel(model):
    """ given a model, save it as a pkl file """
    with open(SAVE_PATH + "pos{}.pkl".format(model.pos), "wb") as out_file:
        # wo do not want to save redundant data, so keys and vals are excluded
        if hasattr(model, "keys"):
            del(model.keys)
            del(model.vals)
        pickle.dump(model, out_file)
    print("model save to", SAVE_PATH + "pos{}.pkl".format(model.pos))


def loadModel(file_name):
    """ given a file anme, load and return the corresponding model from the given file """
    with open(SAVE_PATH + file_name, "rb") as in_file:
        model = pickle.load(in_file, encoding = "uft-8")
    print("{} loaded".format(file_name))
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

        if self.pos != 1:
            self.factor_this_prev = DictionaryFactor([Y[pos - 2], Y[pos - 1]], [n_vars, n_vars])

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
            
            if self.pos > 1:
                prev_target_word = int(target_sentence[self.pos - 1])
                self.factor_this_prev[(prev_target_word, target_word)] += 1

            print("{}\t".format(count), end = "\r")
            count += 1
        print("\ndone")


    def learnTestMode(self):
        for i in range(MAX_LENGTH):
            print(self.factor_list[i].getNonZeroEntryCount())
            self.factor_list[i].learnMode()

        if self.pos > 1:
            self.factor_this_prev.learnMode()


    def learn(self, data_loader):
        # start = time()
        # part_time = 0
        
        print("Prograss:")
        count = 0
        tens_thousand = 0

        self.learnTestMode()

        for sample_batch in data_loader:
            sample_batch["input"] = sample_batch["input"].to("cuda")
            # 10 items in a batch
            for i in range(10):
                # load a pair of data
                input_sentence = sample_batch["input"][i]
                target_sentence = sample_batch["target"][i]

                # get target word and previous target word
                target_word = int(target_sentence[self.pos])
                prev_word = None if self.pos <= 1 else int(target_sentence[self.pos - 1])

                # if count == tens_thousand:
                #     for i in range(MAX_LENGTH):
                #         self.factor_list[i].learnMode()
                #     if self.pos > 1:
                #         self.factor_this_prev.learnMode()
                #     tens_thousand += 10000

                # use this model to generate an output word
                # part_start = time()
                output_word = self.getWordFast(input_sentence[1:-1], prev_word)
                # part_time += time() - part_start

                # if the output word is not target word
                # then P(target word | input word) shoud increase
                # ans P(output word | input word) should drcrease
                # for all factors
                if output_word != target_word:
                    for i in range(MAX_LENGTH):
                        input_word = int(input_sentence[i + 1])
                        self.factor_list[i][(input_word, output_word)] -= 1
                        self.factor_list[i][(input_word, target_word)] += 1

                    if self.pos > 1:
                        self.factor_this_prev[(prev_word, output_word)] -= 1
                        self.factor_this_prev[(prev_word, target_word)] += 1

                #print("total time: {:.5f}, getWord time: {:.5f}".format(time() - start, part_time))
                print("{}/{}\t".format(count,DATA_COUNT), end = "\r")
                count += 1


    def getWord(self, input_sentence, prev_word = None):
        """
        input_sentence is a list of ids!
        prev_word is the previous word (in index form)
        given the input sentence, the function choose and return a word, with its occurances
        """

        ids = input_sentence
        # word_choices in format: {word_index : probability}
        word_choices = dict()
        for i in range(MAX_LENGTH):
            # observe ith word is ids[i]
            # words_probs is a tensor in format:
            # [[word_id_1, word_id_2, word_id_3, ...
            #   prob_1   , prob_2,    prob_3   , ...]]
            temp_factor = self.factor_list[i].copy()
            temp_factor.observe(X[i], ids[i])

            # distance is how far away this word is to the target word
            # further distance may lead to less connection, so less weight
            distance = abs(self.pos - (i+1))
            # iterate all possible words, if the word not in word choices then add it, if exists then update the value
            for k in temp_factor.dictionary:
                # k is the index for a word
                if word_choices.get(k) == None:
                    word_choices[k] = self.c ** distance * self.factor_list[i][k]
                else:
                    word_choices[k] += self.c ** distance * self.factor_list[i][k]

        # given the previous word (observed), get the probability of this word
        if self.pos > 1 and prev_word != None:
            temp_factor = self.factor_this_prev.copy()
            temp_factor.observe(Y[self.pos - 2], prev_word)
            for wc in word_choices:
                if wc in self.factor_this_prev.dictionary:
                    word_choices[wc] += self.factor_this_prev[wc]
        # new word_choices should contain all possible word choices and their probability
        # for this position, we want the maximum one for now
        max_k = list(word_choices.keys())[0]
        for k in word_choices:
            if word_choices[k] > word_choices[max_k]:
                max_k = k
        
        return max_k[0]


    def getWordFast(self, input_sentence, prev_word = None):
        """
        Make use of pytorch
        """
        # word_choices is a tensor: wor_choices[word_index] = probability of word
        word_choices = torch.zeros(30003, device="cuda")
        #total = 0
        for i in range(MAX_LENGTH):
            # observe ith word is ids[i]
            # words_probs is a tensor in format:
            # [[word_id_1, word_id_2, word_id_3, ...
            #   prob_1   , prob_2,    prob_3   , ...]]
            #start = time()
            words, probs = self.factor_list[i].fastObserve(input_sentence[i])
            #total += time() - start
            # distance is how far away this word is to the target word
            # further distance may lead to less connection, so less weight
            distance = abs(self.pos - (i+1))
            # iterate all possible words, if the word not in word choices then add it, if exists then update the value
            if words.size(0) == 0:
                continue
            word_choices[words] += self.c ** distance * probs
        #print(total)

        # given the previous word (observed), get the probability of this word
        if self.pos > 1 and torch.is_tensor(prev_word):
            words, probs = self.factor_this_prev.fastObserve(prev_word)
            if words.size(0) != 0:
                word_choices[words] += probs
        # new word_choices should contain all possible word choices and their probability
        # for this position, we want the maximum one for now
        return torch.argmax(word_choices)


    def getWords(self, input_sentence, num, prev_word = None):
        """
        Similar to getWord, but this returns num words, all of them are possible choices
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
                    word_choices[k] = self.c ** distance * self.factor_list[i][k]
                else:
                    word_choices[k] += self.c ** distance * self.factor_list[i][k]

        # given the previous word (observed), get the probability of this word
        if self.pos > 1 and prev_word != None:
            self.factor_this_prev.observe(Y[self.pos - 2], prev_word)
            for wc in word_choices:
                if wc in self.factor_this_prev.dictionary:
                    word_choices[wc] += 0 * self.factor_this_prev[wc]
        
        # new word_choices should contain all possible word choices and their probability
        # for this position, we want the maximum one for now
        best_choices = sorted(word_choices, key = lambda x : word_choices[x], reverse = True)
        
        return [x[0] for x in best_choices[:num]]


