import json 
import numpy as np
import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence
from new import tokenized, stemmer, bag_of_words
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split



np.set_printoptions(threshold=sys.maxsize)

dreams = pd.read_excel("C:/Users/luca_/Documents/Python/DSProjects/dreams_reports.xlsx")
dreams = pd.DataFrame(dreams[:420])
political_dreams = pd.read_excel("C:/Users/luca_/Documents/Python/DSProjects/political_dreams01.xlsx")

dreams['labels'].replace(to_replace= 'neg', value=0, inplace=True)
dreams['labels'].replace(to_replace= 'pos', value=1, inplace=True)






bags_of_dreams = []
all_words_p = []
for dream in dreams['dreams']:
    token = tokenized(str(dream))
    stem = stemmer(token)
    all_words_p.append(stem)
flat_all_words = [item for i in all_words_p for item in i]

dreams['stem'] = all_words_p

labels = np.array(dreams['labels'])
labels = labels.flatten()

for dream in dreams['stem']:
    bag = bag_of_words(dream, flat_all_words)
    bags_of_dreams.append(bag)

bags_of_dreams = np.array(bags_of_dreams)


labels_flattened = labels.flatten()




X = bags_of_dreams
y = labels_flattened

X_train, X_test, y_train, y_test = train_test_split(X,y)

clsifier = GaussianNB()
clsifier.fit(X_train, y_train)

y_pred = clsifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)





















# for dream in dreams['dreams']:
#     all_words = all_words
#     dream = tokenized(dream)
#     dream = lemmatizer(dream)
#     dream = bag_of_words(dream, all_words)
#     final_dreams.append(dream)

# print(final_dreams[:5])

#     tag = dreams['labels']
#     tags.append(tag)
#     w = utils.tokenize(str(dream))
#     all_words.extend(w)
#     xy.append((w, tag)) 


# ignore = ['?', '!', ".", ","]
# all_words = [utils.lemmatize(w) for w in all_words if w not in ignore]
# # all_words = sorted(set(all_words))
# # tags = sorted(set(tags))


# X_train = []
# y_train = []
# for (pattern_sentence, tag) in xy:
#     bag = utils.bag_of_words(pattern_sentence, all_words)
#     X_train.append(bag)

#     label = tags.index(tag)
#     y_train.append(label)

# X_train = np.array(X_train)
# y_train = np.array(y_train)
# print(X_train[1])
# print(y_train)
