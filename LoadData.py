import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

path = 'D:\Semseter 9\Big Data\CyberBullying\dataset\Dataset.json'
df = pd.read_json(path)
print(df.head)

for i in range(0,len(df)):
    if df.annotation[i]['label'][0] == '1':
        df.annotation[i] = 1
    else:
        df.annotation[i] = 0


df.drop(['extras'],axis = 1,inplace = True)
print(df.head)
df['annotation'].value_counts().sort_index().plot.bar()
plt.show()

#Biasness
print("PosiNon cyber trollingtive: ", df.annotation.value_counts()[0]/len(df.annotation)*100,"%")
print("Cybertrolling: ", df.annotation.value_counts()[1]/len(df.annotation)*100,"%")

