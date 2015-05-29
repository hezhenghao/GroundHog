#!/usr/bin/python2.7

"""
(Comment by HZH)
generate.py: Script that reads text files named "train"， "valid" and "test" under
a path and generate NPZ files containing the vocabulary files and the numberized
text files.

SYNOPSIS
========
Input
    path        path of a folder that contains three text files named "train", "valid" and "test"
    dest        path and filename to save the output file
    level       either "words" or "chars". The level at which the text content is tokenized.
    oov_rate    a floating-point number which represents the out-of-vocabulary rate of the file
                "train" (e.g. if oov_rate = 0.01, then the most frequent tokens that take up a
                combined frequency mass of 99% in the file "train" will be added to the vocabulary).
    dtype       the datatype of the output arrays representing the numberized text files
Output
    file "$dest.npz"
        oov             an integer that is used to represent out-of-vocabulary tokens in the
                        numberized text files
        vocabulary      a dict where keys are tokens and values are the corresponding indices used
                        in the numberization of the text files. A tuple (<unk>: $oov) is added in
                        addition to the in-vocabulary tokens.
        n_$level        size of vocabulary
        train_$level, valid_$level, test_$level
                        numberized text files stored as Numpy arrays, generated from the files
                        "$path/train", "$path/valid", "$path/test", respectively
    file "$dest_dict.npz"
        unique_$level   a dict that is the inverse of $vocabulary, i.e. where keys are indices and
                        values are the corresponding tokens
"""
from collections import Counter
import ConfigParser
import argparse
import os
import time
import sys
import numpy


def construct_vocabulary(dataset, shrink_method = 'oov_rate', threshold = 0.1, level = 'words'):
    if shrink_method not in ['oov', 'size']:
        print 'ERROR: "', shrink_method, '" is not a recognized shrink method.'
    filename = os.path.join(dataset,  'train')
    fd = open(filename, 'rt')
    txt = fd.read()
    if level == 'words':
        txt = txt.replace('\n', ' \n ')
        txt = txt.replace('  ', ' ')
        txt = txt.split(' ')
        txt = [x for x in txt if x != '']
    # Order the words
    print ' .. sorting words'
    all_items = Counter(txt).items()
    no_end = [x for x in all_items if x[0] !='\n']
    freqs = [x for x in all_items if x[0] == '\n'] + \
            sorted(no_end,
                   key=lambda t: t[1],
                   reverse=True)
    print ' .. shrinking the vocabulary size'
    # Decide length
    all_freq = float(sum([x[1] for x in freqs]))
    if shrink_method == 'oov':
        up_to = len(freqs)
        oov = 0.
        remove_word = True
        while remove_word:
            up_to -= 1
            oov += float(freqs[up_to][1])
            if oov / all_freq > threshold:
                remove_word = False
        up_to += 1
    else:
        up_to = int(threshold)
    freqs = freqs[:up_to]
    words = [x[0] for x in freqs]
    vocab = dict(zip(words, range(up_to)))
    freq_rate = [x[1]/all_freq for x in freqs]
    return vocab, freq_rate, freqs


def grab_text(path, filename, vocab, oov_default, dtype, level):
    filename = os.path.join(path, filename)
    fd = open(filename, 'rt')
    txt = fd.read()
    if level == 'words':
        txt = txt.replace('\n', ' \n ')
        txt = txt.replace('  ', ' ')
        txt = txt.split(' ')
        txt = [x for x in txt if x != '']
        return numpy.asarray(
            [vocab.get(w, oov_default) for w in txt],
            dtype=dtype)
    else:
        return numpy.array(
            [vocab.get(w, oov_default) for w in txt],
            dtype=dtype)


def main(parser):
    o = parser.parse_args()
    dataset = o.path
    print 'Constructing the vocabulary ..'
    vocab, freqs, freq_wd = construct_vocabulary(dataset, o.shrink_method, o.threshold, o.level)
    vocab['<unk>'] = numpy.max(list(vocab.values()))+1
    
    oov_default = vocab["<unk>"]
    print "EOL", vocab["\n"]
    print "<unk>", vocab["<unk>"]
    print 'Constructing train set'
    train = grab_text(dataset, 'train', vocab, oov_default, o.dtype, o.level)
    print 'Constructing valid set'
    valid = grab_text(dataset, 'valid', vocab, oov_default, o.dtype, o.level)
    print 'Constructing test set'
    test = grab_text(dataset, 'test', vocab, oov_default, o.dtype, o.level)
    print 'Saving data'

    if o.level == 'words':
        data = {'train_words': train, 'valid_words': valid, 'test_words': test, 'n_words': len(vocab)}
    else:
        data = {'train_chars': train, 'valid_chars': valid, 'test_chars': test, 'n_chars': len(vocab)}
    keys = {'oov': oov_default, 'freqs': numpy.array(freqs), 'vocabulary': vocab, 'freq_wd': freq_wd}
    all_keys = dict(keys.items() + data.items())
    
    numpy.savez(o.dest, **all_keys)
    inv_map = [None] * len(vocab.items())
    for k, v in vocab.items():
        inv_map[v] = k

    if o.level == 'words':
        numpy.savez(o.dest+"_dict", unique_words=inv_map)
    else:
        numpy.savez(o.dest+"_dict", unique_chars=inv_map)
    print '... Done'


def get_parser():
    usage = """
This script generates more numpy friendly format of the dataset from a text
file.  The script will save the entire file into a numpy .npz file. The file
will contain the following fields:

    'train' : array/matrix where each element (word or letter) is
              represented by an index from 0 to vocabulary size or the
              oov value (out of vocabulary). It is the training data.
    'test' : array where each element (word or letter) is represented by an
             index from 0 to vocabulary size or the oov value. This is the
             test value.
    'valid' : array where each element (word or letter) is represented by an
             index from 0 to vocabulary size or the oov value. This is the
             validation set.
    'oov' : The value representing the out of vocabulary word
    'vocab_size' : The size of the vocabulary (this number does not account
                   for oov

FIXME: The current script supports generating a .npz file with either character
sequences or word sequences only.
    """
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('path', 
            default="ntst",
            help=('path to the dataset files: you should have {path}/train, {path}/test and {path}/valid'))
    parser.add_argument('--dest',
                      help=('Where to save the processed dataset (i.e. '
                            'under what name and at what path). It will generate {dest}.npz and {dest}_dict.npz'),
                      default='tmp_data')
    parser.add_argument('--level',
                      help=('Processing level. Either `words` or `chars`. '
                            'If set to word, the result dataset has one '
                            'token per word, otherwise a token per letter'),
                      default='words')
    """
    parser.add_argument('--n-chains',
                      type=int,
                      help=('Number of parallel chains for the training '
                            'data. The way it works, is that it takes the '
                            'training set and divides it in `n_chains` that '
                            'should be processed in parallel by your model'),
                      default=1)
    """
    parser.add_argument('--shrink-method',
                        help=('How to shrink the vocabulary.'
                            'If shrink-method = `size`, the most frequent'
                            '`threshold` tokens are kept in the vocabulary.'
                            'If shrink-method = `oov`, the most infrequent'
                            'tokens that have a combined occurrence rate of'
                            '`threshold` are discarded from the vocabulary.')
                        default='oov')
    parser.add_argument('--threshold',
                      type=float,
                      help=('Meaning varies according to shrink-method.'),
                      default=0.)
    parser.add_argument('--dtype',
                      help='dtype in which to store data',
                      default='int32')
    return parser

if __name__ == '__main__':
    main(get_parser())
