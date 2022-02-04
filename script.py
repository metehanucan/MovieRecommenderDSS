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
