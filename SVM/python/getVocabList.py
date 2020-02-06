def getVocabList():
    """
    Reads the fixed vocabulary list in vocab.txt and returns a dictionary of the words
    """
    vocabList = {}
    vocab = open('vocab.txt').read().split('\n')
    for v in vocab:
        if '\t' in v: idx, word = v.split('\t')
        vocabList[word] = idx
    return vocabList