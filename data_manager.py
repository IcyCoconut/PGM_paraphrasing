import torch
import pickle as pkl # used to read dict.txt file
import codecs # used to read dataset file with utf-8 mode
from string import punctuation

# the path to data folder
DATA_PATH = "data/"
# the maximum length of a sentence
MAX_LENGTH = 15
BOS_ID = 30000
EOS_ID = 30001
UNK_ID = 30002

def wordsToIds(words: str or list) -> torch.tensor:
    """
    Given a sentence, convert each word into its corresponding id
    If the sentence is over 15 words, only translate the first 15 words
    If the sentence is below 15 words, 
    """
    if type(words) == str:
        words = words.split(" ")

    sentence_length = min(MAX_LENGTH, len(words))
    dictionary = getWordDict()

    # we need to add <BOS> and <EOS>, so actually the sequence has length 17
    # initialize the indecies too all <EOS>
    ids = torch.ones(MAX_LENGTH + 2, dtype = torch.int32) * dictionary[0]["<EOS>"]
    ids[0] = dictionary[0]["<BOS>"]

    # the first one is <BOS>, so the 1st element should be ids[i + 1]
    for i in range(sentence_length):
        ids[i + 1] = dictionary[0][words[i]]
    
    return ids


def sentencesToIds(sentences: list) -> torch.tensor:
    """
    Given a list of strings
    Generate a matrix, each element is an index
    each row represents one sentence
    column i is the index of the ith word in each sentence
    """
    n_sentences = len(sentences)
    dictionary = getWordDict()
    # trans_table is used to remove punctuations in a sentence
    # punctuations is converted to spaces
    trans_table = str.maketrans(punctuation, " "*len(punctuation), "")
    # initialize all element to <EOS>
    ids = torch.ones((n_sentences, MAX_LENGTH + 2), dtype = torch.int32) * dictionary[0]["<EOS>"]
    # column index 0 should be all <BOS>
    ids[:, 0] = dictionary[0]["<BOS>"]

    for i in range(n_sentences):
        # this line removes \n at the end of the sentence as well as all punctuations
        # and convert it to lower case
        sentence = sentences[i][:-1].translate(trans_table).lower()
        word_list = sentence.split() # we don't want \n character at the end so [:-1]
        for j in range(min(MAX_LENGTH, len(word_list))):
            idx = dictionary[0].get(word_list[j])
            ids[i, j + 1] = idx if idx != None else dictionary[0]["<UNK>"]
        print("{:.3f}%".format(i/n_sentences*100), end = "\r")

    return ids


def idsToWords(ids: list or torch.tensor) -> list:
    """ Given a sequence of ids, convert each id into corresponding word """
    dictionary = getWordDict()
    word_list = []
    for idx in ids:
        word_list.append(dictionary[1][int(idx)])
    return word_list


def getWordDict() -> dict:
    """
    get the word & index mapping dictionary
    dictionary[0] are mapping from word to its index
    dictionary[1] are mapping from index to the word
    """
    with open(DATA_PATH + "quoradata/dict.pkl", "rb") as data_file:
        d = pkl.load(data_file, encoding = "utf-8")
    d[0]["<BOS>"] = BOS_ID
    d[0]["<EOS>"] = EOS_ID
    d[0]["<UNK>"] = UNK_ID
    d[1][BOS_ID] = "<BOS>"
    d[1][EOS_ID] = "<EOS>"
    d[1][UNK_ID] = "<UNK>"
    return d


class Sentences(torch.utils.data.Dataset):

    def __init__(self, load_mode: str, transform = None):
        """ 
        Prepare quora data sentence dataset
        load_mode can be three values below:
            "train": load train set data
            "validation": load validation set data
            "test": load test set data
        """
        assert (load_mode in ["train", "validation", "test"]), \
            "load_mode can only be train, validation or test"

        if load_mode == "train":
            self.input_path = DATA_PATH + "quoradata/train_pair.txt"
            self.target_path = DATA_PATH + "quoradata/train_label_pair.txt"
        elif load_mode == "validation":
            self.input_path = DATA_PATH + "quoradata/val_pair.txt"
            self.target_path = DATA_PATH + "quoradata/val_label_pair.txt"
        else:
            self.input_path = DATA_PATH + "quoradata/test_pair.txt"
            self.target_path = DATA_PATH + "quoradata/test_label_pair.txt"

        self._loadDataAsIds()
        self.transform = transform


    def _loadDataAsIds(self):
        """
        according to dataset path, load the entire dataset into memory as a huge tensor
        The tensor format please see sentencesToIds comments
        """
        print("Loading dataset ...")
        with codecs.open(self.input_path, "r", encoding="utf-8") as input_file:
            sentences = input_file.readlines()
            self.input_data = sentencesToIds(sentences)
        print("Half loaded ...")

        with codecs.open(self.target_path, "r", encoding="utf-8") as target_file:
            sentences = target_file.readlines()
            self.target_data = sentencesToIds(sentences)

        print("Loading done.")


    def __len__(self):
        """ get the length of the entire dataset """
        return self.input_data.size(0)


    def __getitem__(self, idx):
        """ get a (input, groundtruth) pair """
        sample = {"input": self.input_data[idx], "target": self.target_data[idx]}

        if self.transform:
            sample = self.transform(sample)
        
        return sample


def test():
    train_set = Sentences("train")
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size = 1, 
        shuffle = False, 
    )

    c = 0
    for sample in train_loader:
        c += 1
        data = sample["input"]
        target = sample["target"]
        print(data)
        print(target)
        print(idsToWords(data[0]))
        print(idsToWords(target[0]))
        if c == 10:
            break

if __name__ == "__main__":
    test()