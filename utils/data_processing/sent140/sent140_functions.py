import re
import torch
import numpy as np
from utils.utility_functions.arguments import Arguments
import json


def split_line(line):
    """Split given line/phrase into list of words
    Args:
        line: string representing phrase to be split

    Return:
        list of strings, with each string representing a word
    """
    return re.findall(r"[\w']+|[.,!?;]", line)


def _word_to_index(word, indd):
    """Returns index of given word based on given lookup dictionary
    returns the length of the lookup dictionary if word not found
    Args:
        word: string
        indd: dictionary with string words as keys and int indices as values
    """
    if word in indd:
        return indd[word]
    else:
        return len(indd)


def line_to_indices(line, indd, max_words=25):
    """Converts given phrase into list of word indices

    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer
    representing unknown index to returned list until the list's length is
    max_words
    Args:
        line: string representing phrase/sequence of words
        indd: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list
    Return:
        indl: list of word indices, one index for each word in phrase
    """
    line_list = split_line(line)  # split phrase in words
    indl = []
    for word in line_list:
        cind = _word_to_index(word, indd)
        indl.append(cind)
        if (len(indl) == max_words):
            break
    for i in range(max_words - len(indl)):
        indl.append(len(indd))
    return indl


def process_x(raw_x_batch):
    x_batch = [e[4] for e in raw_x_batch]
    x_batch = [line_to_indices(e, Arguments.word_indices, Arguments.sequence_len) for e in x_batch]
    temp = np.asarray(x_batch)
    x_batch = torch.from_numpy(np.asarray(x_batch))
    return x_batch


def process_y(raw_y_batch):
    return torch.from_numpy(np.asarray(raw_y_batch, dtype=np.float64))


def get_word_emb_arr(path):
    with open(path, 'r') as inf:
        embs = json.load(inf)
    vocab = embs['vocab']
    word_emb_arr = np.array(embs['emba'])
    indd = {}
    for i in range(len(vocab)):
        indd[vocab[i]] = i
    vocab = {w: i for i, w in enumerate(embs['vocab'])}
    return word_emb_arr, indd, vocab


def sent140_preprocess_x(X):
    x_batch = [e[4] for e in X]  # list of lines/phrases
    x_batch = [line_to_indices(e, Arguments.word_indices, 25) for e in x_batch]
    x_batch = torch.from_numpy(np.asarray(x_batch))
    return x_batch


def sent140_preprocess_y(raw_y_batch):
    # return torch.from_numpy(np.asarray(raw_y_batch, dtype=np.float32))
    return torch.from_numpy(np.asarray(raw_y_batch, dtype=np.float64)) / 2

