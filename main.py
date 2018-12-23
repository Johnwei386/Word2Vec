#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from __future__ import division

import argparse
import random
import time
import numpy as np
from config import Config
from helper import Helper
from util import sgd
from word2vec import *

if __name__ == '__main__':
    paser = argparse.ArgumentParser(description='Generating word vector with word2vec')
    paser.add_argument('-d', '--data_path', help="Path of corpus", default="datasets/nlpcc2016")
    paser.add_argument('-t', '--target_path', help="Path to save wordVector", default="datasets/word2vector.txt")
    args = paser.parse_args()
    config = Config(args)
    data_path = args.data_path
    target_path = args.target_path
    dataset = Helper(path=data_path, tablesize=config.tablesize)
    tokens = dataset.get_tokens()
    nWords = len(tokens)
    print("Total {} tokens".format(nWords))

    '''for k,v in sorted(tokens.items(), key=lambda x:x[1]):
        print(k,v)
    print(nWords)'''

    random.seed(config.random_seed)
    np.random.seed(config.np_random_seed)
    startTime = time.time()
    # initialize word vectors
    dimVectors = config.vectors_dim
    wordVectors = np.concatenate(
        ((np.random.rand(nWords, dimVectors) - 0.5) / dimVectors, # centure word vectors
        np.zeros((nWords, dimVectors))), # context word vectors
        axis=0)
    # use sgd train word vectors
    wordVectors = sgd(
        lambda vec: word2vec_sgd_wrapper(word2vecModel = skipgram, 
                                         tokens = tokens, 
                                         wordVectors = vec, 
                                         dataset = dataset, 
                                         C = config.context_size, 
                                         word2vecCostAndGradient = negSamplingCostAndGradient, 
                                         batchsize = config.batchsize),
        x0 = wordVectors, 
        step = config.step, 
        iterations = config.iterations, 
        postprocessing = None, 
        useSaved = config.is_save, 
        PRINT_EVERY = config.PRINT_EVERY,
        ANNEAL_EVERY = config.ANNEAL_EVERY,
        SAVE_PARAMS_EVERY = config.SAVE_PARAMS_EVERY,
        expcost = config.expcost)

    print "training took %d seconds" % (time.time() - startTime)
    wordVectors = np.concatenate(
        (wordVectors[:nWords,:], wordVectors[nWords:,:]),
        axis=1)
    print("WordVectors shape: {} x {}".format(wordVectors.shape[0], wordVectors.shape[1]))


