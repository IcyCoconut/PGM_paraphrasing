# PGM_paraphrasing

## Useful functions and classes:
### There are other functions or classes, but only important and frequently used ones are listed below
### data_manager.py : A file contains some class and function used to manage data
- wordsToIds: convert sentence to index sequence
- sentencesToIds: convert many sentences to a matrix of index
- idsToWords: convert index sequence to sentence
- getWordDict: get 2 dictionaries, you can use it to convert words and index back and forth
- Sentences: a dataset class
### PGM_basics.py : A file defines some basic classes of Probabilistic Graphical Models
- DictionaryFactor: a factor class, it uses a dictionary to store only non-zero entries of a factor
- HmmFactor: factor designed for HMM, make use of dictionary when feeding data into the factor, but can use fixed() to change dictionary to tensors, which enable fast execution.
### model.py : Defines some model
- saveModel: save a model given file name
- loadModel: load and return a model given file name
- PartialModel: a class which defines part of a model, in case that the entire model is too large (not final version, may change in the future)
- HiddenMarkovModel: a class defines a hidden markov model, after learnDataset(), it can generate output sentence given input sentence
### train_and_test.py : Contains some functions that do trainings or testings
- loadDataset: load a dataset and return the dataset loader, you can use a for loop to get each data from the loader
- writeLog: print current date and time with a message, also write the same line to log.txt file
- buildModel: build and save a complete model (not HMM for now, it will change in the future)
- learnModel: learn a model (not HMM for now, it will change in the future)
- test: test a model (not HMM, so will change)
- hmm_train_save: train and save a HMM
- test_hmm: using 20000 test set sentences, and get 20000 output sentences, save the result
