import pandas as pd 
import matplotlib.pyplot as plt
from gen_wordcloud import *

#read data
python = pd.read_excel('../data/Github_python.xlsx')
java = pd.read_excel('../data/Github_java.xlsx')
C = pd.read_excel('../data/Github_C.xlsx')
java_s = pd.read_excel('../data/Github_javascript.xlsx')
Cpp = pd.read_excel('../data/Github_Cpp.xlsx')
C_sharp = pd.read_excel('../data/Github_C#.xlsx')
#add a language column
python['Language'] = 'Python'
java['Language'] = 'Java'
C['Language'] = 'C'
java_s['Language'] = 'JavaScript'
Cpp['Language'] = 'C++'
C_sharp['Language'] = 'C#'
#concat to one DataFrame
data = pd.concat([python, java, C, java_s, Cpp, C_sharp], axis = 0, ignore_index = True)
data.to_excel('../data/Github_all.xlsx')
data = data.drop(['Unnamed: 0'], axis = 1)
data = data.set_index('Full Name')
#select rows with stats > 10000
data = data[data['Stars'] >= 10000]
data = data.sort_values(by = 'Stars', ascending = False)

x  = data.groupby('Language').apply(len)
x.plot(kind = 'pie', subplots = True, autopct = '%.2f')
plt.title('Pie Chart')
plt.savefig('../picture/pie.png')
plt.show()
#wordcloud = gen_wordcloud(data)
#plt.imshow(wordcloud)
#plt.axis('off')
#plt.show()
#wordcloud.to_file('../picture/wordcloud_all.jpg')


'''
year = [x[:4] for x in python['Created Time']]
month = [x[5:7] for x in python['Created Time']]
python['year'] = year
python['month'] = month
python = python.sort_values(by = ['year', 'month'])
plt.scatter(python['year']+ '-'+ python['month'],python['Stars'])
plt.scatter(python['Forks'],python['Stars'])
plt.xticks(rotation = 90)
plt.show()
plt.scatter(python['Size'],python['Stars'])
plt.show()
'''
