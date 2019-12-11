import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np
import string
import re
from wordcloud import WordCloud
from imageio import imread
import matplotlib.pyplot as plt


def gen_wordcloud(name):
    text = ""
    for item in name['Description'].dropna():
        text += item
    for item in name['Full Name'].dropna():
        text += item
    word = nltk.word_tokenize(text)
    word = [x.lower() for x in word]
    word = [i for i in word if (i not in stopwords.words('english')) and (i not in string.punctuation)]
    word = nltk.pos_tag(word)
    newlist = []
    p = re.compile('^N')
    for w, flag in word:
        if p.search(flag) != None:
            newlist.append(w)
    word_dic = dict()
    for k,v in nltk.FreqDist(newlist).items():
        word_dic[str(k)] = v
    img = imread('../picture/github.png')
    wordcloud = WordCloud(font_path = '../simhei.ttf', background_color = 'white', mask = img, max_words = 80).generate_from_frequencies(word_dic)
    return wordcloud
if __name__ == '__main__':
    python = pd.read_excel('../data/Github_python.xlsx')
    java = pd.read_excel('../data/Github_java.xlsx')
    javas = pd.read_excel('../data/Github_javascript.xlsx')
    C = pd.read_excel('../data/Github_C.xlsx')
    Cpp = pd.read_excel('../data/Github_Cpp.xlsx')
    C_sharp = pd.read_excel('../data/Github_C#.xlsx')
    plt.figure(figsize = (20,20))
    plt.subplot(2,3,1)
    plt.imshow(gen_wordcloud(python))
    plt.axis('off')
    plt.title('Python')
    plt.subplot(2,3,2)
    plt.imshow(gen_wordcloud(java))
    plt.axis('off')
    plt.title('Java')
    plt.subplot(2,3,3)
    plt.imshow(gen_wordcloud(javas))
    plt.axis('off')
    plt.title('JavaScript')
    plt.subplot(2,3,4)
    plt.imshow(gen_wordcloud(C))
    plt.axis('off')
    plt.title('C')
    plt.subplot(2,3,5)
    plt.imshow(gen_wordcloud(Cpp))
    plt.axis('off')
    plt.title('C++')
    plt.subplot(2,3,6)
    plt.imshow(gen_wordcloud(C_sharp))
    plt.axis('off')
    plt.title('C#')
    plt.show()
