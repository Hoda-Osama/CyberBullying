import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
import nltk
import re, string
from nltk.corpus import stopwords
import LoadData
df =LoadData.df

def Stemming():
    #Stemming
    porter_stemmer = PorterStemmer()
    #punctuations
    nltk.download('punkt')
    tok_list = []
    size = df.shape[0]

    for i in range(size):
        word_data = df['content_without_puncs'][i]
        nltk_tokens = nltk.word_tokenize(word_data)
        final = ''
        for w in nltk_tokens:
            final = final + ' ' + porter_stemmer.stem(w)
        tok_list.append(final)

    df['content_tokenize'] = tok_list
    del df['content_without_puncs']
    print(df)

def toknize():
    noNums = []
    for i in range(len(df)):
        noNums.append(''.join([i for i in df['content_tokenize'][i] if not i.isdigit()]))

    df['content'] = noNums
    print(df)

def ftfIdf():
    tfIdfVectorizer=TfidfVectorizer(use_idf=True, sublinear_tf=True)
    tfIdf = tfIdfVectorizer.fit_transform(df.content.tolist())
    print(tfIdf)
    print(tfIdf.shape) # means total rows  20001 with 14783 features
    df2 = pd.DataFrame(tfIdf[2].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"]) #for second entry only(just to check if working)
    df2 = df2.sort_values('TF-IDF', ascending=False)
    print (df2.head(10))
    dfx = pd.DataFrame(tfIdf.toarray(), columns = tfIdfVectorizer.get_feature_names())
    print(dfx)
    return tfIdfVectorizer, tfIdf

def display_scores(vectorizer, tfidf_result):
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    i=0
    for item in sorted_scores:
        print ("{0:50} Score: {1}".format(item[0], item[1]))
        i = i+1
        if (i > 25):
          break    


nltk.download('stopwords') 
stop = set(stopwords.words('english'))

regex = re.compile('[%s]' % re.escape(string.punctuation))

def test_re(s):
    return regex.sub('', s)

df ['content_without_stopwords'] = df['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df ['content_without_puncs'] = df['content_without_stopwords'].apply(lambda x: regex.sub('',x))
del df['content_without_stopwords']
del df['content']
print(df)

Stemming()
toknize()
tfIdfVectorizer,tfIdf =ftfIdf()
#top 25 words
display_scores(tfIdfVectorizer, tfIdf)
X=tfIdf.toarray()
y = np.array(df.annotation.tolist())

#Spltting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


#Training data biasness
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

#Test Data
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

#Random oversampling on training data
from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy='not majority')
X_over, y_over = oversample.fit_resample(X_train, y_train)

print(X_over.shape)
print(y_over.shape)

unique_elements, counts_elements = np.unique(y_over, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))