#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import numpy as np 
import random
from util import sigmoid, sigmoid_grad, softmax

def normalizeRows(x):
    """ Row normalization function
    """
    denom = np.apply_along_axis(lambda x: np.sqrt(np.dot(x, x.T)), 1, x)
    x /= denom
    return x

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word,期望得到的单词下标
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors
    """
    #  Calculate the predictions:
    vhat = predicted 
    z = np.dot(outputVectors, vhat)
    preds = softmax(z)

    #  Calculate the cost:
    cost = -np.log(preds[target])

    # Gradients
    z = preds.copy()
    z[target] -= 1.0

    grad = np.outer(z, vhat)
    gradPred = np.dot(outputVectors.T, z) 
    return cost, gradPred, grad

def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """
    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))
    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)
    cost = 0
    z = sigmoid(np.dot(outputVectors[target], predicted))

    cost -= np.log(z)
    grad[target] += predicted * (z - 1.0)
    gradPred += outputVectors[target] * (z - 1.0)

    for k in range(K):
        samp = indices[k + 1]
        z = sigmoid(np.dot(outputVectors[samp], predicted))
        cost -= np.log(1.0 - z)
        grad[samp] += predicted * z
        gradPred += outputVectors[samp] * z
    return cost, gradPred, grad

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.
    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    cword_idx = tokens[currentWord]
    vhat = inputVectors[cword_idx]

    for j in contextWords:
        u_idx = tokens[j]
        c_cost, c_grad_in, c_grad_out = word2vecCostAndGradient(vhat, u_idx, outputVectors, dataset)
        cost += c_cost
        gradIn[cword_idx] += c_grad_in
        gradOut += c_grad_out
    return cost, gradIn, gradOut

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient, batchsize = 50):
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N / 2, :]
    outputVectors = wordVectors[N / 2:, :]
    for i in range(batchsize):
        C1 = random.randint(1, C)
        centerword, context = dataset.getRandomContext(C1)
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        c, gin, gout = word2vecModel(centerword,
                                     C1, 
                                     context, 
                                     tokens, 
                                     inputVectors, 
                                     outputVectors,
                                     dataset, 
                                     word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N / 2, :] += gin / batchsize / denom
        grad[N / 2:, :] += gout / batchsize / denom
    return cost, grad