#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import os
import random
import logging
import numpy as np
import cPickle as pickle

logger = logging.getLogger('word2vector')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Helper:
    def __init__(self, path=None, tablesize=100000):
        if path is None or not os.path.exists(path):
            path = "datasets/nlpcc2016"
        
        self.path = path
        self.tablesize = tablesize
        self.token2idx = dict()
        self.token2freq = dict()
        self.wordcount = 0
        self.idx2token = []
        self.sentences = []
        self.sentlengths = 0
        self.cumsentlen = 0
        self.sampleTable = None
        self.rejectProb = None
        self.allsentences = None

        self.sentences = self.get_sentences()
        self.token2idx = self.get_tokens()

    def get_sentences(self):
        with open(self.path, 'r') as f:
            sentence = []
            for line in f:
                line = line.strip().split()
                if len(line) == 0:
                    self.sentences.append(sentence)
                    sentence = []
                else:
                    sentence.append(line[0].decode('utf-8'))
        #logger.info("Load %d sentence", len(self.sentences))
        self.sentlengths = np.array([len(s) for s in self.sentences])
        self.cumsentlen = np.cumsum(self.sentlengths) # cumulative sum along axis
        return self.sentences

    def get_tokens(self):
        idx = 0
        for sentence in self.get_sentences():
            for word in sentence:
                self.wordcount += 1
                if not word in self.token2idx:
                    self.token2idx[word] = idx
                    self.idx2token += [word]
                    self.token2freq[word] = 1
                    idx += 1
                else:
                    self.token2freq[word] += 1
        UNK = "UNK".decode('utf-8')
        self.token2idx[UNK] = idx
        self.idx2token += [UNK]
        self.token2freq[UNK] = 1
        self.wordcount += 1
        #logging.info("Load %d tokens", len(self.token2idx))

        return self.token2idx

    def get_rejectProb(self):
        if self.rejectProb is not None:
            return self.rejectProb
        threshold = 1e-5 * self.wordcount
        nTokens = len(self.get_tokens())
        rejectProb = np.zeros((nTokens,))
        for i in range(nTokens):
            w = self.idx2token[i]
            freq = 1.0 * self.token2freq[w]
            rejectProb[i] = max(0, 1 - np.sqrt(threshold / freq))
        self.rejectProb = rejectProb
        #logger.info("Generate reject rate for every token")
        return self.rejectProb

    def get_allSentences(self):
        if self.allsentences:
            return self.allsentences
        sentences = self.get_sentences()
        rejectProb = self.get_rejectProb()
        tokens = self.get_tokens()
        # first duplicate sentences to 30 copies and then sample word from this big sentences sets
        # every sentence in sentences will be random sampling 30 times
        allsentences = [[w for w in s if 0 >= rejectProb[tokens[w]] or np.random.random() >= rejectProb[tokens[w]]]
                        for s in sentences * 30]
        allsentences = [s for s in allsentences if len(s) > 1]
        # size equal 30 times size of sentences, but erevry sentence in allsentences just contain some random sampled word
        self.allsentences = allsentences 
        logger.info("Duplicate sentence set to generate a big set")
        return self.allsentences

    def get_sampleTable(self):
        #logger.info("Generate sample table...")
        if self.sampleTable is not None:
            return self.sampleTable
        nTokens = len(self.get_tokens())
        samplingFreq = np.zeros((nTokens,))
        self.get_allSentences()
        i = 0
        for w in range(nTokens):
            w = self.idx2token[i]
            if w in self.token2freq:
                freq = 1.0 * self.token2freq[w]
                freq = freq ** 0.75
            else:
                freq = 0.0
            samplingFreq[i] = freq
            i += 1
        samplingFreq /= np.sum(samplingFreq)
        samplingFreq = np.cumsum(samplingFreq) * self.tablesize
        self.sampleTable = [0] * self.tablesize
        j = 0
        for i in range(self.tablesize):
            while i > samplingFreq[j]:
                j += 1
            self.sampleTable[i] = j
        #logger.info("Generate sample table done.")
        return self.sampleTable

    def sampleTokenIdx(self):
        return self.get_sampleTable()[np.random.randint(0, self.tablesize - 1)]

    def getRandomContext(self, C=5):
        # logger.info("Achieve a random context")
        allsent = self.get_allSentences() # Expand the base range of the sample
        sentID = random.randint(0, len(allsent) - 1) # one random number
        sent = allsent[sentID]
        wordID = random.randint(0, len(sent) - 1) 
        context = sent[max(0, wordID - C):wordID] # left side context of centure word 
        if wordID+1 < len(sent): # right side context of centure word
            context += sent[wordID+1:min(len(sent), wordID + C + 1)]
        centerword = sent[wordID] 
        context = [w for w in context if w != centerword]

        if len(context) > 0:
            return centerword, context
        else:
            return self.getRandomContext(C)


if __name__ == '__main__':
    helper = Helper()
    #helper.get_allSentences()
    helper.getRandomContext()
