from collections import Counter
from data.Dependency import *
import numpy as np


class CharVocab(object):
    PAD, UNK = 0, 1
    def __init__(self, char_counter, min_occur_count=2):
        self._id2char = ['<pad>', '<unk>']

        for char, count in char_counter.most_common():
            if count > min_occur_count:
                self._id2char.append(char)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._char2id = reverse(self._id2char)
        if len(self._char2id) != len(self._id2char):
            print("serious bug: chars dumplicated, please check!")

        print("Vocab info: #chars %d" % (self.char_size))

    def char2id(self, xs):
        if isinstance(xs, list):
            return [self._char2id.get(x, self.UNK) for x in xs]
        return self._char2id.get(xs, self.UNK)

    def id2char(self, xs):
        if isinstance(xs, list):
            return [self._id2char[x] for x in xs]
        return self._id2char[xs]

    @property
    def char_size(self):
        return len(self._id2char)

def createCharVocab(corpusFile, min_occur_count):
    char_counter = Counter()
    assert isinstance(corpusFile, list)
    print(corpusFile)
    for path in corpusFile:
        print(path)
        with open(path, 'r', encoding="utf8") as infile:
            for sentence in readDepTree(infile):
                for dep in sentence:
                    for char in dep.form:
                        char_counter[char] += 1
    return CharVocab(char_counter, min_occur_count)
