#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 08:50:10 2018

@author: angus
"""

# libraries
import tweepy
from tweepy import OAuthHandler
import pandas as pd
from tqdm import tqdm
import pymongo
from pymongo import MongoClient
import pprint
import os



# setup api variable
os.chdir('/home/angus/projects/project_templates/20180922_NLPTemplate/src')
import passwords

auth = OAuthHandler(passwords.consumer_key, passwords.consumer_secret)
auth.set_access_token(passwords.access_token, passwords.access_secret)
 
api = tweepy.API(auth, wait_on_rate_limit=True)

user=api.me()
print(user.name)




# test out iterating over some tweets
for tweet in tweepy.Cursor(api.search,q='bet').items(2):
    try:
        user_id = tweet.user.id
        username = tweet.user.screen_name
        message = tweet.text
        print(tweet)
    
    except tweepy.TweepError as e:
        print(e.reason)
    except StopIteration:
        break




# add in tweet_mode='extended' to get full text
for tweet in tweepy.Cursor(api.search,q='bet',tweet_mode='extended').items(1):
    testTweet=tweet

testTweet.text
testTweet.full_text
testTweet.retweeted_status.full_text


# create dataframe with some tweets
tweetsOutput = pd.DataFrame(columns=['user_id','username','text'])
for tweet in tweepy.Cursor(api.search,q='bet',tweet_mode='extended').items(100):
    try:
        user_id = tweet.user.id
        username = tweet.user.screen_name
        try:
            message = tweet.retweeted_status.full_text
        except:
            try:
                message = tweet.full_text
            except:
                    message = tweet.text
        
        df = pd.DataFrame([[user_id,username,message]], columns = ['user_id','username','text'])
        tweetsOutput = tweetsOutput.append(df, ignore_index=True)
    
    except tweepy.TweepError as e:
        print(e.reason)
    except StopIteration:
        break

tweetsOutput.shape
tweetsOutput.iloc[201,2]
    



# setup connection with mongoDB
client = MongoClient('mongodb://localhost:27017')
db=client.tweets

# test input
test_insert = {'user_id': str(testTweet.user.id),
               'username': testTweet.user.screen_name,
               'text': testTweet.full_text
               }
db.betTweets.insert_one(test_insert)

# insert already obtained tweets
for r in range(tweetsOutput.shape[0]):
    tweet_insert = {'user_id': str(tweetsOutput.iloc[r,0]),
                    'username': tweetsOutput.iloc[r,1],
                    'text': tweetsOutput.iloc[r,2]
                    }
    db.betTweets.insert_one(tweet_insert)

# get new tweets and insert directly
for tweet in tqdm(tweepy.Cursor(api.search,q='bet',tweet_mode='extended').items(2000)):
    try:
        if hasattr(tweet, 'retweeted_status'):
            message = tweet.retweeted_status.full_text
        elif hasattr(tweet, 'full_text'):
            message = tweet.full_text
        else:
            message = tweet.text
        
        tweet_insert = {'user_id': str(tweet.user.id),
                        'username': tweet.user.screen_name,
                        'text': message
                        }
        
        db.betTweets.insert_one(tweet_insert)
    
    except tweepy.TweepError as e:
        print(e.reason)
    except StopIteration:
        break




# get tweets from db
for tweet in db.betTweets.find():
    df=pd.DataFrame([[tweet['user_id'],tweet['username'],tweet['text']]], columns=['user_id','username','text'])
    tweetsOutput=tweetsOutput.append(df, ignore_index=True)









