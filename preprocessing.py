import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


def load_datasets(full = False):

    pos_path = "data/twitter-datasets/train_neg.txt"
    neg_path = "data/twitter-datasets/train_pos.txt"
    if full:
        pos_path = "data/twitter-datasets/train_neg_full.txt"
        neg_path = "data/twitter-datasets/train_pos_full.txt"
    df_train_neg = pd.read_csv(neg_path, delimiter="\t", header=None, names = ['tweets'], on_bad_lines="skip")
    df_train_pos = pd.read_csv(pos_path, delimiter="\t", header=None, names = ['tweets'], on_bad_lines="skip")
    df_train_neg["label"] = -1
    df_train_pos["label"] = 1

    df_train = pd.concat([df_train_pos,df_train_neg])

    df_test = pd.read_csv("data/twitter-datasets/test_data.txt", delimiter="\t", header=None, names = ['tweets'], on_bad_lines="skip")
    df_test["tweets"] = df_test["tweets"].apply(lambda row: row.split(",",2)[1])

    return df_train, df_test

def remove_tags(df):
    df_cleaned = df.copy()
    df_cleaned['tweets'] = df_cleaned['tweets'].apply(lambda tweet: re.sub(r'<.*?>', '', tweet).strip())
    return df_cleaned

def tokenize_and_preprocess(df, stop_words = False, stemming = False, lemmatization = False):
    df_cleaned = df.copy()
    df_cleaned['tokens'] = df_cleaned['tweets'].apply(lambda tweet: word_tokenize(tweet))
    # remove stop words
    if stop_words:
        stop_words = stopwords.words('english')
        df_cleaned['tokens'] = df_cleaned['tokens'].apply(lambda tokens: [token for token in tokens if token.lower() not in stop_words])
    # stemming
    if stemming:
        ps = PorterStemmer()
        df_cleaned['tokens'] = df_cleaned['tokens'].apply(lambda tokens: [ps.stem(token) for token in tokens])
    # lemmatization
    if lemmatization:
        wordnet_lemmatizer = WordNetLemmatizer()
        df_cleaned['tokens'] = df_cleaned['tokens'].apply(lambda tokens: [wordnet_lemmatizer.lemmatize(token) for token in tokens])
    # remove the tweets columns
    df_cleaned.drop(['tweets'], axis=1, inplace=True)
    df_cleaned = df_cleaned.reindex(columns=['tokens', 'label'])
    return df_cleaned  

