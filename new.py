import nltk
import re
from part_of_speech import get_part_of_speech
from nltk.util import ngrams
from collections import Counter
import numpy as np


def tokenized(report):
    cleaned = re.sub('\W+', ' ', report)
    tokenized = nltk.word_tokenize(cleaned)
    return tokenized

def stemmer(report):
    stemmer = nltk.PorterStemmer()
    stemmed = [stemmer.stem(token) for token in report]
    return stemmed

def lemmatizer(report):
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token, get_part_of_speech(token)) for token in report]
    return lemmatized

def ngram(report):
    bigram = ngrams(report, 2)
    bigram_count = Counter(bigram)
    return bigram_count


def bag_of_words(report, all_words):
    tokenized_sentence = [w for w in report]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[index] = 1.0
    return bag



