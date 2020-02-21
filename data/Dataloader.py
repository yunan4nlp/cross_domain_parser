from data.Vocab import *
import numpy as np
import torch
from torch.autograd import Variable

def domain_labeling(data, is_source=True):
    new_data = []
    for sent in data:
        if is_source:
            new_data.append([sent, 1])
        else:
            new_data.append([sent, 0])
    return new_data



def read_corpus(file_path, vocab=None):
    data = []
    with open(file_path, 'r', encoding="utf8") as infile:
        for sentence in readDepTree(infile, vocab):
            data.append(sentence)
    return data

def sentences_numberize(sentences, vocab, char_vocab, ignoreTree):
    for sentence in sentences:
        yield sentence2id(sentence, vocab, char_vocab, ignoreTree)

def sentence2id(sentence, vocab, char_vocab, ignoreTree):
    result = []
    for dep in sentence[0]:
        wordid = vocab.word2id(dep.form)
        charid = []
        for char in dep.form:
            charid.append(char_vocab.char2id(char))
        extwordid = vocab.extword2id(dep.form)
        tagid = vocab.tag2id(dep.tag)
        if ignoreTree:
            head = -1
            relid = -1
        else:
            head = dep.head
            relid = vocab.rel2id(dep.rel)
        score = dep.score
        result.append([wordid, extwordid, tagid, head, relid, score, charid])
    return result, sentence[1]


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def batch_data_variable(batch, vocab, char_vocab, ignoreTree=False):
    length = len(batch[0][0])
    batch_size = len(batch)
    for b in range(1, batch_size):
        tree = batch[b][0]
        if len(tree) > length: length = len(tree)

    char_length = -1
    for b in range(batch_size):
        tree = batch[b][0]
        for idx in range(len(tree)):
            l = len(tree[idx].form)
            if l > char_length: char_length = l
    if char_length > 8: char_length = 8

    words = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    extwords = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    tags = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    masks = Variable(torch.Tensor(batch_size, length).zero_(), requires_grad=False)
    scores = Variable(torch.FloatTensor(batch_size, length).zero_(), requires_grad=False)
    domain_labels = Variable(torch.LongTensor(batch_size).zero_(), requires_grad=False)
    chars = Variable(torch.LongTensor(batch_size, length, char_length).zero_(), requires_grad=False)
    char_masks = Variable(torch.Tensor(batch_size, length, char_length).zero_(), requires_grad=False)
    heads = []
    rels = []
    lengths = []
    b = 0
    for info in sentences_numberize(batch, vocab, char_vocab, ignoreTree):
        index = 0
        sentence = info[0]
        domain_labels[b] = info[1]
        length = len(sentence)
        lengths.append(length)
        head = np.zeros((length), dtype=np.int32)
        rel = np.zeros((length), dtype=np.int32)
        # score = np.zeros((length), dtype=np.float32)
        for dep in sentence:
            words[b, index] = dep[0]
            extwords[b, index] = dep[1]
            if dep[2] == None:
                dep[2] = 0
            tags[b, index] = dep[2]
            head[index] = dep[3]
            rel[index] = dep[4]
            scores[b, index] = dep[5]
            masks[b, index] = 1

            if char_length > len(dep[6]):
                max_len = len(dep[6])
            else:
                max_len = char_length

            for char_index in range(max_len):
                chars[b, index, char_index] = dep[6][char_index]
                char_masks[b, index, char_index] = 1
            index += 1
        b += 1
        heads.append(head)
        rels.append(rel)
        # scores.append(score)
    return words, extwords, chars, tags, heads, rels, lengths, masks, char_masks, scores, domain_labels

def batch_variable_depTree(trees, heads, rels, lengths, vocab, arcs_values=None):
    if arcs_values == None:
        for tree, head, rel, length in zip(trees, heads, rels, lengths):
            tree = tree[0]
            sentence = []
            for idx in range(length):
                sentence.append(
                    Dependency(idx, tree[idx].org_form, tree[idx].tag, head[idx], vocab.id2rel(rel[idx])))
            yield sentence
    else:
        for tree, head, rel, length, arc_v in zip(trees, heads, rels, lengths, arcs_values):
            tree = tree[0]
            sentence = []
            for idx in range(length):
                sentence.append(
                    Dependency(idx, tree[idx].org_form, tree[idx].tag, head[idx], vocab.id2rel(rel[idx]), arc_v[idx])
                )
            yield sentence
