import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import joblib

data = pd.read_excel('../data/Github_python.xlsx').dropna(subset = ['Description'])
names = [name for name in data['Full Name']]
descrips = [descrip for descrip in data['Description']]
kmeans = joblib.load('cluster_doc_2.pkl')
clusters = kmeans.labels_.tolist()
df = pd.DataFrame({'Name':names, 'Description':descrips, 'cluster':clusters}, index = [clusters])
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


#visualizing
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
cluster_names = {0: '0',
                 1: '1',
                 2: '2',
                 3: '3',
                 4: '4'}

