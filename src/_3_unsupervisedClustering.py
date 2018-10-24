#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 17:06:41 2018

@author: angus
"""

from pymongo import MongoClient
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from itertools import compress
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

import re
import lda



'''
GET TWEETS FROM DB
'''
# setup connection with mongoDB
client = MongoClient('mongodb://localhost:27017')
db=client.tweets

# read tweets
tweetsOutput = []
for tweet in tqdm(db.betTweets.find()):
    tweetsOutput.append([tweet['user_id'],tweet['username'],tweet['text']])
    
#tweetsDF=pd.DataFrame(tweetsOutput, columns=['user_id','username','text'])
tweetsNP=np.array(tweetsOutput)




'''
PROCESS TWEETS TO DTM
'''
# remove duplicates
tweetsNonDuplicated = list(set(tweetsNP[:,2]))

# process
os.chdir('/home/angus/projects/project_templates/20180922_NLPTemplate/src')
import _2_textPreprocessing as tp

#tweetsProcessed=tp.text_preprocessor(tweetsNonDuplicated, ngram_max_len=3)

countvec = CountVectorizer(strip_accents='ascii',
                           preprocessor=tp.dtm_preprocessor,
                           tokenizer=tp.dtm_tokenizer,
                           stop_words=stopwords.words('english'),
                           ngram_range=(1,3),
                           analyzer='word',
                           max_df=1.0,
                           min_df=10 # float -> proportion, integer -> absolute number
                           )

tfidfvec = TfidfVectorizer(strip_accents='ascii',
                           preprocessor=tp.dtm_preprocessor,
                           tokenizer=tp.dtm_tokenizer,
                           stop_words=stopwords.words('english'),
                           ngram_range=(1,3),
                           analyzer='word',
                           max_df=1.0,
                           min_df=10, # float -> proportion, integer -> absolute number
                           use_idf=True)

dtm = countvec.fit_transform(tweetsNonDuplicated)
tfidf = tfidfvec.fit_transform(tweetsNonDuplicated)
vocab = countvec.get_feature_names()
vocab = tfidfvec.get_feature_names()

# nb with min_df as 1 vocab is len 130000, with 10 vocab len is 1197

# have a look at dtm
dtmdense=dtm.todense()

tweetsNonDuplicated[1]
testlist=list(np.concatenate((dtmdense[1,:]>0).tolist()))
list(compress(vocab, testlist))


'''
K-MEANS CLUSTERING
'''
# compare a broad range of ks to start
ks = [2, 10, 20, 50, 100, 500]

# track a couple of metrics
sil_scores = []
inertias = []

# fit the models, save the evaluation metrics from each run
for k in ks:
    logging.warning('fitting model for {} clusters'.format(k))
    model = KMeans(n_clusters=k, n_jobs=-1, random_state=123)
    model.fit(tfidf)
    labels = model.labels_
    sil_scores.append(silhouette_score(tfidf, labels))
    inertias.append(model.inertia_) # inertia = within cluster sum of squares

# plot the quality metrics for inspection
fig, ax = plt.subplots(2, 1, sharex=True)

plt.subplot(211)
plt.plot(ks, inertias, 'o--')
plt.ylabel('inertia')
plt.title('kmeans parameter search')

plt.subplot(212)
plt.plot(ks, sil_scores, 'o--')
plt.ylabel('silhouette score')
plt.xlabel('k');

# go with 50 for now
best_k = 50
km_mod=KMeans(n_clusters=best_k, n_jobs=-1, random_state=123)
km_mod.fit(tfidf)


# how many tweets are in each cluster?
plt.bar(range(len(set(km_mod.labels_))), np.bincount(km_mod.labels_))

plt.ylabel('population')
plt.xlabel('cluster label')
plt.title('k={} cluster populations'.format(best_k));

# truncating the axis again!
plt.ylim(0,3000);

list(compress(tweetsNonDuplicated, km_mod.labels_==6))



'''
LDA
'''
# fitting the collapsed gibbs lda model
K = 10
alpha = 0.1
eta = 0.01
collapsed_gibbs = lda.LDA(n_topics=K, n_iter=100, alpha=alpha, eta=eta, random_state=123)
collapsed_gibbs.fit(dtm)

# getting the document topic probabilities
doc_topics = collapsed_gibbs.doc_topic_
topic_words = collapsed_gibbs.topic_word_
log_likelihoods = collapsed_gibbs.loglikelihoods_
doc_topics = collapsed_gibbs.doc_topic_

n_top_words = 25

topic_top_words = []
for t in range(K):
    topic_top_words.append((np.array(vocab)[np.argsort(topic_words[t,:])[::-1][:n_top_words]]).tolist())


# look at some examples
n_samples=10
topic=1
np.array(tweetsNonDuplicated)[(np.argsort(doc_topics[:,topic])[:-(n_samples+1):-1])]


# predict the topics and review output


























