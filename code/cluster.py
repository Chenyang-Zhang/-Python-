
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import mpld3

def tokenize_and_stem(text):
    tokens = [word for sentence in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sentence)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize(text):
    tokens = [word.lower() for sentence in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sentence)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

data = pd.read_excel('../data/Github_python.xlsx').dropna(subset = ['Description'])
names = [name for name in data['Full Name']]
descrips = [descrip for descrip in data['Description']]
stemmer = SnowballStemmer('english')
stopwords = nltk.corpus.stopwords.words('english')

total_stemmed = []
total_tokenized = []
for item in descrips:
    allwords_stemmed = tokenize_and_stem(item)
    total_stemmed.extend(allwords_stemmed)

    allwords_tokenized = tokenize(item)
    total_tokenized.extend(allwords_tokenized)

vocab_df = pd.DataFrame({'words': total_tokenized}, index = total_stemmed)
#print(vocab_df.head())
tfidf_vectorizer = TfidfVectorizer(max_df = 0.7, max_features = 200000, min_df = 15, stop_words = stopwords,
        use_idf = True, tokenizer = tokenize_and_stem, ngram_range = (1, 3))
tfidf_matrix = tfidf_vectorizer.fit_transform(descrips)
#print(tfidf_matrix.shape)
dist = 1 - cosine_similarity(tfidf_matrix)
terms = tfidf_vectorizer.get_feature_names()

#K-Means
#kmeans = KMeans(n_clusters = 5)
#kmeans.fit(tfidf_matrix)
#clusters = kmeans.labels_.tolist()
#joblib.dump(kmeans, 'cluster_doc_2.pkl')
kmeans = joblib.load('cluster_doc_2.pkl') #read trained file
clusters = kmeans.labels_.tolist()
df = pd.DataFrame({'Name':names, 'Description':descrips, 'cluster':clusters}, index = [clusters])
'''
print(df['cluster'].value_counts())
print('Top terms per Cluster:')
order = kmeans.cluster_centers_.argsort()[:, ::-1]
for i in range(5):
    print('Cluster %d top terms:' % i)
    for j in order[i, :5]:
        print(' %s' % vocab_df.loc[terms[j].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end = ',')
    print()

    print('Cluster %d repo description:' % i)
    num = 1
    for name in df.loc[i]['Description'].values.tolist():
        print('%d: %s' % (num, name[:150]))
        num += 1
        if num == 21:
            break
    print('=============================================================')
'''

#visualizing
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
cluster_names = {0: '0',
                 1: '1',
                 2: '2',
                 3: '3',
                 4: '4'}







