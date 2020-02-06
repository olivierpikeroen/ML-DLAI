import numpy as np 
from getVocabList import *

def emailFeatures(word_indices):
    """
    Produces a feature vector from the word indices 
    """
    vocabList = getVocabList()
    return [1 if vocabList[word] in word_indices else 0 for word in vocabList] 