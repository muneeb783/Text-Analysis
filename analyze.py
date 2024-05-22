text_file = open("pg1400.txt", 'r', encoding = 'utf-8')
great_expect = text_file.read()

#print(great_expect)

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

from gensim.corpora import Dictionary
from gensim.models import ldamodel
from gensim.models.coherencemodel import CoherenceModel
from wordcloud import WordCloud

import pandas as pd 
import numpy as np
import random 
import re
import matplotlib.pyplot as plt

word_cloud_text = great_expect.lower()
word_cloud_text = re.sub("[^a-zA-Z0-9]", " ", word_cloud_text)

tokens = word_tokenize(word_cloud_text)
tokens = (word for word in tokens if word not in stopwords.words('english'))
tokens = (word for word in tokens if len(word) >= 3)

#DATA CLEANING

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"

text = " " + great_expect + "  "
text = text.replace("\n"," ")
text = re.sub(prefixes,"\\1<prd>",text)
text = re.sub(websites,"<prd>\\1",text)
text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
if "..." in text: text = text.replace("...","<prd><prd><prd>")
if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
if "”" in text: text = text.replace(".”","”.")
if "\"" in text: text = text.replace(".\"","\".")
if "!" in text: text = text.replace("!\"","\"!")
if "?" in text: text = text.replace("?\"","\"?")
text = text.replace(".",".<stop>")
text = text.replace("?","?<stop>")
text = text.replace("!","!<stop>")
text = text.replace("<prd>",".")
sentences = text.split("<stop>")
sentences = [s.strip() for s in sentences]
sentences = pd.DataFrame(sentences)
sentences.columns = ['sentence']

#print(len(sentences))
#print(sentences.head(10))

sentences.drop(sentences.index[:59], inplace = True)
sentences = sentences.reset_index(drop = True)
#print(sentences.head(10))

stpowards_wc = set(stopwords.words('english'))
wordCloud = WordCloud(max_words=100, stopwords=stpowards_wc, random_state=1).generate(word_cloud_text)

'''
plt.figure(figsize=(12,16))
plt.imshow(wordCloud)
plt.axis('off')
plt.show()
'''

fdist = nltk.FreqDist(tokens)
#print(fdist.most_common(50))

'''
plt.figure(figsize=(12,6))
fdist.plot(50)
plt.show()
'''

analyzer = SentimentIntensityAnalyzer()
sentences['compound'] = [analyzer.polarity_scores(x)['compound'] for x in sentences['sentence']]
sentences['neg'] = [analyzer.polarity_scores(x)['neg'] for x in sentences['sentence']]
sentences['neu'] = [analyzer.polarity_scores(x)['neu'] for x in sentences['sentence']]
sentences['pos'] = [analyzer.polarity_scores(x)['pos'] for x in sentences['sentence']]

#print(sentences.head(10))

positive_sent = sentences.loc[sentences['compound'] > 0]
negative_sent = sentences.loc[sentences['compound'] < 0]
neutral_sent = sentences.loc[sentences['compound'] == 0]

'''
print(sentences.shape)
print(len(positive_sent))
print(len(negative_sent))
print(len(neutral_sent))
'''

#plt.figure(figsize=(14,6))
#plt.hist(sentences['compound'], bins = 50);

data = sentences['sentence'].values.tolist()

def text_processing(texts):
    # Remove numbers and alphanumerical words we don't need
    texts = [re.sub("[^a-zA-Z]+", " ", str(text)) for text in texts]
    # Tokenize & lowercase each word
    texts = [[word for word in text.lower().split()] for text in texts]
    # Stem each word
    lmtzr = WordNetLemmatizer()
    texts = [[lmtzr.lemmatize(word) for word in text] for text in texts]
    # Remove stopwords
    stoplist = stopwords.words('english')
    texts = [[word for word in text if word not in stoplist] for text in texts]
    # Remove short words less than 3 letters in length
    texts = [[word for word in tokens if len(word) >= 3] for tokens in texts]
    return texts

data = text_processing(data)
dictionary = Dictionary(data)

corpus = [dictionary.doc2bow(text) for text in data]

np.random.seed(1)
k_range = range(6,20,2)
scores = []
for k in k_range:
    ldaModel = ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=20)
    cm = CoherenceModel(model=ldaModel, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    #print(cm.get_coherence())
    scores.append(cm.get_coherence())

"""
plt.figure()
plt.plot(k_range, scores)
plt.show()
"""

model = ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=6, passes=20)


