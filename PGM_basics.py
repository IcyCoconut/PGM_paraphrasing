import torch
import numpy as np
from time import time

MAX_LINES = 10 # maximumly 10 lines are displayed when print a factor
INT_DTYPE = torch.long # datatype for integer tensors
FLOAT_DTYPE = torch.float64 # datatype for float tensors

class Factor():
    """
    This factor stores entries using a table
    It stores a probability for every possible assignment of variables
    """
    def __init__(self, var: list, card: list):
        assert (len(var) == len(card)), "variable and cardinality size mismatch"
        self.var = torch.tensor(var) if type(var) == list else var.clone()
        self.card = torch.tensor(card)
        self.val = torch.zeros(torch.prod(self.card))


    def __str__(self) -> str:
        """
        Return a table representation of this factor
        If the factor is too big, then only show the first 10 lines
        """
        string = "Factor with {} variables:\n".format(self.var.size(0))
        counter = torch.zeros_like(self.card)

        # this part generate the title of the table
        for v in self.var:
            string += "{:<8}|".format(int(v))
        string += "values\n" + "-"*((self.card.size(0) + 1) * 8) + "\n"

        # every iteration of this while loop generate a line
        val_idx = 0
        while val_idx < min(MAX_LINES, self.val.size(0)):
            for x in counter:
                string += "{:<8}|".format(str(int(x)))
            string += "{:.5f}\n".format(float(self.val[val_idx]))
            counter[-1] += 1
            for i in range(counter.size(0) - 1, -1, -1):
                if counter[i] == self.card[i]:
                    counter[i] -= self.card[i]
                    counter[i - 1] += 1
            val_idx += 1

        if MAX_LINES < self.val.size(0):
            string += "...\n"

        return string


    def __len__(self):
        """ This will return the total count of entries """
        return self.val.size(0)


    def __setitem__(self, key: int or list, value : float):
        """
        Set a value when use factor[key] = value
        If key is an int, then key-th entry is set to value
        If key is a list, then key is treated as the assignment to variables, the corresponding entry will have value
        Example:
            >>> print(f)

                Factor with 2 variables:
                1       |2       |values
                ------------------------
                0       |0       |0.0
                0       |1       |0.0
                1       |0       |0.0
                1       |1       |0.0

            >>> f[3] = 1.5
            >>> f[[1, 0]] = 1.25
            >>> print(f)

                Factor with 2 variables:
                1       |2       |values
                ------------------------
                0       |0       |0.0
                0       |1       |0.0
                1       |0       |1.25
                1       |1       |1.5

        """
        try:
            value = float(value)
        except:
            assert "value must be a number"
        if type(key) == int:
            self.val[key] = value
        else:
            self.val[self.assignmentToIndex(key)] = value


    def __getitem__(self, key: int or list) -> float:
        """
        Get an entry from this factor
        The key follows the same rule as in __setitem__
        """
        if type(key) == int:
            return self.val[key]
        else:
            return self.val[self.assignmentToIndex(key)]


    def assignmentToIndex(self, assignment: list) -> int:
        """ Convert an assignment of variables to index of value (index of the entry) """

        assert(len(assignment) == self.var.size(0)), \
                "size of key({}) must equal to the number of variables({})"\
                .format(assignment, self.var)
        index = assignment[-1]
        for i in range(len(assignment) - 1):
            index += assignment[i] * torch.prod(self.card[i+1:])
        return index


    def indexToAssignment(self, index: int) -> list:
        """ Convert an index of value to as assignment of variables """

        assert(index < self.val.size(0)), "index must less than the number of values"
        assignment = []

        for i in range(len(self.card) - 1):
            divisor = int(torch.prod(self.card[i+1:]))
            assignment.append(index // divisor)
            index %= int(divisor)
        assignment.append(index)
        return torch.tensor(assignment, dtype = INT_DTYPE)


    def __mul__(self, other):
        """
        Do factor product, if f1 and f2 are two factors
        if 
        """
        # prepare new variable and new cardinality
        all_var = torch.cat((self.var, other.var))
        all_card = torch.cat((self.card, other.card))
        new_var = torch.unique(all_var)
        all_var = list(all_var)
        new_card = []
        for v in new_var:
            idx = all_var.index(v)
            new_card.append(all_card[idx])

        result = Factor(new_var, new_card)
        # join_var are variabels that both factors have
        join_var = np.intersect1d(self.var, other.var)
        # if in result factor we have var [a,b,c,d] with assignment [0,1,2,3]
        # if self.var = [b,a,c], assignment[mapA] will be [1, 0, 2]
        # if other.var = [d,a], assignment[mapB] will be [3, 0]
        mapA = torch.zeros(self.var.size(0), dtype = INT_DTYPE)
        mapB = torch.zeros(other.var.size(0), dtype = INT_DTYPE)
        for i in range(result.var.size(0)):
            if result.var[i] in self.var:
                idx = torch.where(self.var == result.var[i])[0]
                mapA[idx] = i
            if result.var[i] in other.var:
                idx = torch.where(other.var == result.var[i])[0]
                mapB[idx] = i

        for i in range(len(result)):
            assignment = result.indexToAssignment(i)
            result[i] = self[assignment[mapA]] * other[assignment[mapB]]

        return result



class DictionaryFactor():
    """
    This factor design uses a dictionary to store assignments:values pairs
    However, it does not store those entries that has 0 probability
    """
    def __init__(self, var: list, card: list):
        assert (len(var) == len(card)), "variable and cardinality size mismatch"
        self.var = torch.tensor(var) if type(var) == list else var.clone()
        self.card = torch.tensor(card)
        self.dictionary = dict()

    
    def __str__(self):
        """ Returns a string representation of this factor """
        string = "Dictionary Factor with {} variables\n".format(self.var.size(0))

        # this part generate the title of the table
        for v in self.var:
            string += "{:<8}|".format(int(v))
        string += "values\n" + "-"*((self.card.size(0) + 1) * 8) + "\n"

        count = 0
        for key, value in self.dictionary.items():
            for x in key:
                string += "{:<8}|".format(str(int(x)))
            string += "{:.5f}\n".format(float(value))
            count += 1
            if count == MAX_LINES:
                string += "...\n"
                break

        return string


    def __len__(self):
        """ Note that this returns the total number of entries """
        return torch.prod(self.card)


    def getNonZeroEntryCount(self):
        """ This method returns the number of non zero entries """
        return len(self.dictionary)


    def __setitem__(self, key: tuple or int, value):
        """
        set an item according to key, key must be a tuple or int
        value can be float or int or tensor
        """
        if value == 0:
            return

        assert (type(key) in [tuple, int]), "key must be a tuple or int"

        if type(key) == int:
            # if key is a single value tensor, also find the assignment
            key = self.indexToAssignment(key)

        self.dictionary[key] = value


    def __getitem__(self, key: tuple or int):
        """
        Get an entry from this factor, given a key as the assignment to variables
        key must be a tuple
        """
        assert (type(key) in [tuple, int]), "key must be a tuple or int"

        if type(key) == int:
            # if key is a single value tensor, also find the assignment
            key = self.indexToAssignment(key)

        value = self.dictionary.get(key)
        return (0 if value == None else value)


    def __mul__(self, other):
        """
        Do factor product, if f1 and f2 are two factors
        if 
        """
        # prepare new variable and new cardinality
        all_var = torch.cat((self.var, other.var))
        all_card = torch.cat((self.card, other.card))
        new_var = torch.unique(all_var)
        all_var = list(all_var)
        new_card = []
        for v in new_var:
            idx = all_var.index(v)
            new_card.append(all_card[idx])

        result = DictionaryFactor(new_var, new_card)
        # join_var are variabels that both factors have
        join_var = np.intersect1d(self.var, other.var)
        # if in result factor we have var [a,b,c,d] with assignment [0,1,2,3]
        # if self.var = [b,a,c], assignment[mapA] will be [1, 0, 2]
        # if other.var = [d,a], assignment[mapB] will be [3, 0]
        mapA = torch.zeros(self.var.size(0), dtype = INT_DTYPE)
        mapB = torch.zeros(other.var.size(0), dtype = INT_DTYPE)
        for i in range(result.var.size(0)):
            if result.var[i] in self.var:
                idx = torch.where(self.var == result.var[i])[0]
                mapA[idx] = i
            if result.var[i] in other.var:
                idx = torch.where(other.var == result.var[i])[0]
                mapB[idx] = i

        for i in range(len(result)):
            assignment = torch.tensor(result.indexToAssignment(i), dtype = INT_DTYPE)

            # wee need to convert key to tuple
            assignment_A = tuple([int(x) for x in assignment[mapA]])
            assignment_B = tuple([int(x) for x in assignment[mapB]])
            tuple_assignment = tuple([int(x) for x in assignment])

            value = self[assignment_A] * other[assignment_B]

            if value != 0:
                result[tuple_assignment] = value

        return result


    def copy(self):
        """ return a copy of self """
        copy_of_self = DictionaryFactor(self.var.clone(), self.card.clone())
        copy_of_self.dictionary = self.dictionary.copy()
        return copy_of_self


    def indexToAssignment(self, index: int) -> tuple:
        """
        Convert an index of value to as assignment of variables
        Returns the assignment as a tensor
        """

        assert(index < self.__len__()), "index must less than the number of values"
        assignment = []

        for i in range(len(self.card) - 1):
            divisor = int(torch.prod(self.card[i+1:]))
            assignment.append(index // divisor)
            index %= int(divisor)
        assignment.append(index)
        return tuple(assignment)


    def observe(self, observed_var, assignment):
        """
        Observe a variable
        observed_var is the observed variable
        assignment is the observed value of this variable
        Example: factor.observe(1, 3) means variable 1 has assignment 3
        """
        
        var_idx = torch.where(self.var == observed_var)[0]

        # if the original var is [1,2,3,4] and card is [2,3,2,3]
        # observe (2, 1), then the new var is [1,3,4] and card is [2,2,3]
        self.var = torch.cat([self.var[:var_idx], self.var[var_idx+1:]])
        self.card = torch.cat([self.card[:var_idx], self.card[var_idx+1:]])

        # following the previous comment, if there is an entry [0, 1, 0, 2] = 0.25
        # then we observe variable 2 with value 1, we add a new entry [0, 0, 2] = 0.25
        # and delete the old one. If there is an entry [1, 0, 1, 2] = 0.13, we do not 
        # add any new entries, and directly remove this entry, becaue variable 2 is 0, not 1

        new_dict = dict()
        for key in self.dictionary:
            if key[var_idx] == assignment:
                new_dict[key[:var_idx] + key[var_idx+1:]] = self.dictionary[key]
                #new_dict[(key[0],)] = self.dictionary[key]
        self.dictionary = new_dict


    def fastObserve(self, assignment):
        """
        NOTE: this only work when where are exactly 2 variables
        Observe the first variable with given value assignment
        Returns a tensor in format:
            [[word_idx, prob], [word_idx, prob], ...]
            actually a single variable factor
        Example:
            f = DictionaryFactor([4,5], [2,2]) with values [1 2 3 4]
            in table:
                4   5   v
                ---------
                0   0   1
                0   1   2
                1   0   3
                1   1   4
            factor.observe(1) means variable 4 has assignment 1
            result will be tensor([[0, 3], [1, 4]])
        
        This is the special designd version for this task
        the speed is around 17.5 times faster than the observe function above
        """
        assignment = torch.tensor(assignment, dtype=torch.long)
        keys = torch.tensor(list(self.dictionary.keys()), dtype = torch.long)
        vals = torch.tensor(list(self.dictionary.values()))
        remain_idx = torch.where(keys[:, 0] == assignment)[0]

        # if the original var is [1,2] and card is [2,3]
        # then the new var is [2] and card is [3]
        result = torch.cat((keys[remain_idx, 1].view(1, -1), vals[remain_idx].view(1, -1)))
        return result.view(-1, 2)
        
    def normalize(self):
        """ Normalize the factor, make all entries sum to 1 """
        total = sum(self.dictionary.values())
        for k in self.dictionary:
            self.dictionary[k] /= total





def basicTest():
    # Factor class basic test
    test = Factor([2, 1], [3, 2])   # test __init__
    test[[1,0]] = 1.25  # test __setitem__(key is list)
    test[3] = 1.5   # test  __setitem__(key is int)
    print(test) # test __str__
    print(test[2])  # test __getitem__

    # Dictionary factor basic test
    test1 = DictionaryFactor([1, 2, 3], [2, 2, 3]) # test __init__
    test1[(1,0,1)] = 1.25   # test __setitem__
    print(test1)    # test __str__
    print(test1[(1,0,1)])   # test __getitem__
    print(test1[(0, 1, 2)])


def productTest():
    # define 2 factors
    f1 = Factor([2,1,3], [2,2,2])
    f2 = Factor([4,1], [2,2])
    # fill them with values
    for i in range(len(f1)):
        f1[i] = i/10
    for j in range(len(f2)):
        f2[j] = j/10
    # print them and print the factor product
    print(f1)
    print(f2)
    print(f1*f2)

    # do same thing to dictionary factor
    f1 = DictionaryFactor([2,1,3], [2,2,2])
    f2 = DictionaryFactor([4,1], [2,2])
    for i in range(len(f1)):
        f1[f1.indexToAssignment(i)] = i/10
    for j in range(len(f2)):
        f2[f2.indexToAssignment(j)] = j/10
    print(f1)
    print(f2)
    print(f1*f2)


def obsNormTest():
    # test observe and normalize
    f = DictionaryFactor([7, 8], [2, 3])
    for i in range(6):
        f[i] = i
    print(f)
    f.observe(8, 1)
    print(f)
    f.normalize()
    print(f)

    f = DictionaryFactor([2,6,4], [2,2,2])
    for i in range(8):
        f[i] = 1+i
    print(f)
    f.observe(6, 1)
    print(f)
    f.normalize()
    print(f)


def designedObserveSpeedTest():
    f1 = DictionaryFactor([2,3], [30000, 30000])
    for i in range(30000):
        f1[i] = i
    print(f1)

    start = time()
    f2 = f1.fastObserve(1)
    ckpt_1 = time()
    #print(f2)
    #print(dict(f2))
    f1.observe(2, 1)
    print("fast: {}\nslow: {}".format(ckpt_1 - start, time() - ckpt_1))

    
if __name__ == "__main__":
    # basicTest()
    # productTest()
    # obsNormTest()
    designedObserveSpeedTest()
