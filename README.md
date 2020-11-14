# PGM_paraphrasing

## Code Overview:
### data_manager.py : A file contains some class and function used to manage data
- wordsToIds: convert sentence to index sequence
- sentencesToIds: convert many sentences to a matrix of index
- idsToWords: convert index sequence to sentence
- getWordDict: get 2 dictionaries, you can use it to convert words and index back and forth
- Sentences: a dataset class
- test: used to test the dataset class, you can see here to learn how to use Sentences dataset
### PGM_basics.py : A file defines some basic classes of Probabilistic Graphical Models
- Factor: a factor class, defines a factor, it use a list to store all entries of a factor
- DictionaryFactor: another factor class, it uses a dictionary to store only non-zero entries of a factor