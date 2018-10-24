#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 15:28:08 2018

@author: angus
"""

# libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import pymongo
from pymongo import MongoClient
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.util import everygrams
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import string
from collections import Counter
import re



## get tweets from db
## setup connection with mongoDB
#client = MongoClient('mongodb://localhost:27017')
#db=client.tweets
#
## read tweets
#tweetsOutput = []
#for tweet in tqdm(db.betTweets.find()):
#    tweetsOutput.append([tweet['user_id'],tweet['username'],tweet['text']])
#    
#tweetsDF=pd.DataFrame(tweetsOutput, columns=['user_id','username','text'])
#tweetsNP=np.array(tweetsOutput)




# text pre-processing
def get_all_tokens(tweet_list):
    """
    Helper function to generate a list of text tokens from concatenating
    all of the text contained in Tweets in `tweet_list`
    """
    # concat entire corpus
    all_text = ' '.join((tweet_list))
    # tokenize
    tokens = (TweetTokenizer(preserve_case=False,
                            reduce_len=True, # Replace repeated character sequences of length 3 or greater with sequences of length 3
                            strip_handles=True) # Remove Twitter username handles from text
              .tokenize(all_text))
    # remove symbol-only tokens for now
    tokens = [tok for tok in tokens if not tok in string.punctuation]
    return tokens

#tokens = get_all_tokens(tweetsNP[:,2])
#token_lens = ([len(token) for token in tokens])
#tokens[token_lens.index(max(token_lens))]
#
#print('total number of tokens: {}'.format(len(tokens)))
#
#top_grams = Counter(everygrams(tokens, min_len=2, max_len=4))
#
#top_grams.most_common(25)




# some weird common n-grams so check for duplicates
# get ngrams for every tweet
def get_tweet_ngrams(tweet_list, min_len=1, max_len=3):
    tweet_grams=[]
    for tweet in tweet_list:
        tweet_grams.append(
                list(everygrams(
                        TweetTokenizer(preserve_case=False,
                                       reduce_len=True,
                                       strip_handles=True)
                        .tokenize(tweet), min_len=1, max_len=3))
                )
    return(tweet_grams)

#tweet_grams = get_tweet_ngrams(tweetsNP[:,2])




def get_tweets_with_ngram(tweet_list, ngram_to_find):
    tweet_ngrams=get_tweet_ngrams(tweet_list, min_len=len(ngram_to_find), max_len=len(ngram_to_find))
    tweets_with_ngram=[]
    for t in range(len(tweet_list)):
        if ngram_to_find in tweet_ngrams[t]:
            tweets_with_ngram.append([t,tweet_list[t]])
    return(tweets_with_ngram)

#tweets_with_ngram=get_tweets_with_ngram(tweetsNP[:,2], ('every', 'country', 'in'))
#tweetsNP[[int(x) for x in np.array(tweets_with_ngram)[:,0]],:]
# looks like lots of users who have retweeted the same tweet - remove duplicates
# want to do this for training phase, but not for prediction phase




# remove duplicates
#tweetsNonDuplicated = list(set(tweetsNP[:,2]))


# text preprocessing auxiliary functions (to be combined into one function afterwards)
def replace_urls(corpus, replacement=None):
    """Replace URLs in strings. See also: ``bit.ly/PyURLre``

    Args:
        in_string (str): string to filter
        replacement (str or None): replacment text. defaults to '<-URL->'

    Returns:
        str
    """
    replacement = '<-URL->' if replacement is None else replacement
    pattern = re.compile('(https?://)?(\w*[.]\w+)+([/?=&]+\w+)*')
    return [re.sub(pattern, replacement, doc) for doc in corpus]


def split_into_tokens(corpus, preserve_case=False, remove_handles=True, reduce_length=True):
    # function splits a text string into its individual words/tokens
    # remove_handles is for removing twitter handles
    # reduce_length replaces repeated character sequences of length 3 or greater with sequences of length 3
    #text = unicode(text, 'utf8') - convert bytes into proper unicode (normally not needed)
    #if remove_punct:
    #    #corpus = [re.sub(string.punctuation, "", doc) for doc in corpus]
    #    corpus = [re.sub("[^a-z A-Z]+", "", doc) for doc in corpus]
    return [TweetTokenizer(preserve_case=preserve_case,
                           strip_handles=remove_handles,
                           reduce_len=reduce_length).tokenize(doc) for doc in corpus]
    
    ### additional way of removing punctuation
    #corpus = [doc.translate(None, string.punctuation) for doc in corpus]
    
    ### additional tokenizers below
    #return word_tokenize(testTweet['text'])
    #return wordpunct_tokenize(testTweet['text'])


def remove_stop_words_and_punctuation(tokens, remove_stopwords=True, remove_punctuation=True):
    # function removes stop words and punctuation from list of tokens
    stops = stopwords.words('english') if remove_stopwords else []
    punctuation = list(string.punctuation) if remove_punctuation else []
    removals = stops + punctuation
    filtered_tokens = [item for item in tokens if item not in removals]
    return filtered_tokens


def stemming_words(tokens):
    # function stems words using the Porter Stemmer
    p_stemmer = PorterStemmer()
    return [p_stemmer.stem(i) for i in tokens]


# text preprocessing function
def text_preprocessor(corpus,
                      switch_urls=True,
                      url_replacement=None,
                      preserve_case=False,
                      remove_handles=True,
                      reduce_length=True,
                      remove_stopwords=True,
                      remove_punctuation=True,
                      stem_words=True,
                      ngram_min_len=1,
                      ngram_max_len=1):
    
    # remove urls
    if switch_urls:
        processed_corpus = replace_urls(corpus, replacement=url_replacement)
    else:
        processed_corpus = corpus
    
    # tokenize
    processed_corpus = split_into_tokens(processed_corpus,
                                         preserve_case=preserve_case,
                                         remove_handles=remove_handles,
                                         reduce_length=reduce_length)
    
    # remove stopwords and punctuation
    processed_corpus = [remove_stop_words_and_punctuation(doc,
                                                         remove_stopwords=remove_stopwords,
                                                         remove_punctuation=remove_punctuation)
                        for doc in processed_corpus]
    
    # stem words (NOTE: if stemming then cases will not be preserved)
    if stem_words:
        processed_corpus = [stemming_words(doc) for doc in processed_corpus]
    
    # create all ngrams
    if ngram_max_len > 1 :
        processed_corpus=[list(everygrams(tokens, min_len=ngram_min_len, max_len=ngram_max_len))
                            for tokens in processed_corpus]
    
    return processed_corpus



# dtm generator
def dtm_preprocessor(doc, url_replacement=None, remove_punctuation=True):
    """
    Replace URLs and remove punctuation
    """
    replacement = '<-URL->' if url_replacement is None else url_replacement
    pattern = re.compile('(https?://)?(\w*[.]\w+)+([/?=&]+\w+)*')
    doc = re.sub(pattern, replacement, doc)
    
    filtered_doc = re.sub("[^a-z A-Z]+", "", doc)
    
    return filtered_doc


def dtm_tokenizer(doc, preserve_case=False, remove_handles=True, reduce_length=True):
    return TweetTokenizer(preserve_case=preserve_case,
                          strip_handles=remove_handles,
                          reduce_len=reduce_length).tokenize(doc)

def dtm_creator(corpus,
                strip_accents='ascii',
                preprocessor=dtm_preprocessor,
                tokenizer=dtm_tokenizer,
                stop_words=stopwords.words('english'),
                ngram_range=(1,3),
                analyzer='word',
                max_df=1.0,
                min_df=1):
    
    countvec = CountVectorizer(strip_accents=strip_accents,
                               preprocessor=preprocessor,
                               tokenizer=tokenizer,
                               stop_words=stop_words,
                               ngram_range=ngram_range,
                               analyzer=analyzer,
                               max_df=max_df,
                               min_df=min_df)
    
    return countvec(corpus)







































