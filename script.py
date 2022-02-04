from multiprocessing import Pool
from pymongo import MongoClient
import numpy as np
import pandas as pd
from os import path
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def _connect_mongo(db):
    conn = MongoClient('mongodb+srv://scraper:123qwe@mis463.2chrl.mongodb.net/test?retryWrites=true&w=majority')
    return conn[db]

def read_mongo(db, collection, query={}, no_id=True):
    """ Read from Mongo and Store into DataFrame """

    # Connect to MongoDB
    db = _connect_mongo(db)

    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)

    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df

df = read_mongo(db='test',collection='movies',query={})
df['description'] = df['overview'] + df['title'] + df['genres'].fillna('').str.join(' ')  + 
df['cast'].fillna('').str.join(' ') + 
df['crew'].apply(lambda d: d if isinstance(d, list) else []).apply(lambda x: list(filter(lambda a: a['job'] == 'Director', x))).apply(lambda d: list(map(lambda c: c['name'],d))).str.join(' ')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
stemmer = PorterStemmer()
def stem_words(words_list, stemmer):
  return [stemmer.stem(word) for word in words_list]

def tokenizer(text):
  letters_only = re.sub("[^a-zA-Z]",' ', text)
  words = letters_only.lower().split()
  stopwords_eng = set(stopwords.words("english"))
  useless_words = ['film','movie','available','theater','directed']
  stopwords_eng.update(useless_words)
  useful_words = [x for x in words if not x in stopwords_eng]
  useful_words_string = ' '.join(useful_words)
  tokens = nltk.word_tokenize(useful_words_string)
  stems = stem_words(tokens, stemmer)
  return stems
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(
    tokenizer = tokenizer,
    stop_words = 'english'
)
tfidf_matrix = tfidf.fit_transform(df['description'])
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['id']).drop_duplicates()
import pickle
with open('data.pkl', 'wb') as f:
    pickle.dump(cosine_sim, f)
    
with open('indices.pkl', 'wb') as f:
    pickle.dump(indices, f)
