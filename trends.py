import nltk, re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from classifier import dreams, political_dreams
from new import tokenized, stemmer, lemmatizer, ngram
from nltk.corpus import stopwords
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt



file_dreams = open('dreams.txt', 'w', encoding='utf-8')
file_read= open('dreams.txt', 'r', encoding='utf-8')
for dream in dreams['dreams']:
    file_dreams.write(str(dream))
    file_dreams.write(' ')
for report in political_dreams['dreams']:
    file_dreams.write(str(report))
    file_dreams.write(' ')


words_pre = list(file_read.read().split())
words_pre = lemmatizer(words_pre)


stop_words = stopwords.words('english')


words = []
for word in words_pre:
    if word not in stop_words:
        words.append(word.lower())

words_str = " ".join(s for s in words)
bag = CountVectorizer()
words_bag = bag.fit_transform(words)

tfidf_creator = TfidfVectorizer()
tfidf = tfidf_creator.fit_transform(words)

lda_bag_of_words_creator = LatentDirichletAllocation(learning_method='online', n_components=10)
lda_bag_of_words = lda_bag_of_words_creator.fit_transform(words_bag)

lda_tfidf_creator = LatentDirichletAllocation(learning_method='online', n_components=10)
lda_tfidf = lda_tfidf_creator.fit_transform(tfidf)

print("MAIN TOPICS FOUND")
for topic_id, topic in enumerate(lda_bag_of_words_creator.components_):
    message = "Topic #{}: ".format(topic_id + 1)
    message += " ".join([bag.get_feature_names()[i] for i in topic.argsort()[:-5 :-1]])
    print(message)


wordcloud = WordCloud(background_color='black').generate(words_str)
fig, ax = plt.subplots(figsize=(10,6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.set_axis_off()

plt.imshow(wordcloud)
wordcloud.to_file('dreams_wordcloud.png')




