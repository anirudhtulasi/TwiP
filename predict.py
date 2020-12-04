import tweepy
import sys
import os
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
import numpy as np
import string
import pandas as pd
import pickle
import csv

from unidecode import unidecode
from itertools import islice
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
from collections import Counter

ckey = 'YvQZ4wcEhYo4F4Lj8PQqj500d'
csecret = 'nDMvzbvQxt2zW9GZOalnOikeDcwGwPwvi2USOD5phJCmw5e9wD'
atoken = '994417184322433025-anaAEWzZErK5h8CGTxEgJMPFwfRoMNA'
asecret = 'M2RwOR7CaWHMO7d83iWuKhHmEtgyaomobBNt9bsJhSF3i'
auth = tweepy.OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
api = tweepy.API(auth)

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    # URLs
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')',
                       re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$',
                         re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(
            token) else token.lower() for token in tokens]
    return tokens


def preproc(s):
    # s=emoji_pattern.sub(r'', s) # no emoji
    s = unidecode(s)
    POSTagger = preprocess(s)
    # print(POSTagger)

    tweet = ' '.join(POSTagger)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    #filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in POSTagger:
        if w not in stop_words:
            filtered_sentence.append(w)
    # print(word_tokens)
    # print(filtered_sentence)
    stemmed_sentence = []
    stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
    for w in filtered_sentence:
        stemmed_sentence.append(stemmer2.stem(w))
    # print(stemmed_sentence)

    temp = ' '.join(c for c in stemmed_sentence if c not in string.punctuation)
    preProcessed = temp.split(" ")
    final = []
    for i in preProcessed:
        if i not in final:
            if i.isdigit():
                pass
            else:
                if 'http' not in i:
                    final.append(i)
    temp1 = ' '.join(c for c in final)
    # print(preProcessed)
    return temp1


def getTweets(user):
    csvFile = open('user.csv', 'a', newline='')
    csvWriter = csv.writer(csvFile)
    try:
        for i in range(0, 4):
            tweets = api.user_timeline(
                screen_name=user, count=1000, include_rts=True, page=i)
            for status in tweets:
                tw = preproc(status.text)
                if tw.find(" ") == -1:
                    tw = "blank"
                csvWriter.writerow([tw])
    except tweepy.TweepError:
        print("Failed to run the command on that user, Skipping...")
    csvFile.close()

def classify_user(username):
    # username = input("Please Enter Twitter Account handle: ")
    getTweets(username)
    with open('user.csv', 'rt') as f:
        csvReader = csv.reader(f)
        tweetList = [rows[0] for rows in csvReader]
    os.remove('user.csv')
    with open('CSV_Data/newfrequency300.csv', 'rt') as f:
        csvReader = csv.reader(f)
        mydict = {rows[1]: int(rows[0]) for rows in csvReader}

    vectorizer = TfidfVectorizer(vocabulary=mydict, min_df=1)
    x = vectorizer.fit_transform(tweetList).toarray()
    df = pd.DataFrame(x)


    model_IE = pickle.load(open("Pickle_Data/BNIEFinal.sav", 'rb'))
    model_SN = pickle.load(open("Pickle_Data/BNSNFinal.sav", 'rb'))
    model_TF = pickle.load(open('Pickle_Data/BNTFFinal.sav', 'rb'))
    model_PJ = pickle.load(open('Pickle_Data/BNPJFinal.sav', 'rb'))

    answer = []
    IE = model_IE.predict(df)
    SN = model_SN.predict(df)
    TF = model_TF.predict(df)
    PJ = model_PJ.predict(df)


    b = Counter(IE)
    value = b.most_common(1)
    # print(value)
    if value[0][0] == 1.0:
        answer.append("I")
    else:
        answer.append("E")

    b = Counter(SN)
    value = b.most_common(1)
    # print(value)
    if value[0][0] == 1.0:
        answer.append("S")
    else:
        answer.append("N")

    b = Counter(TF)
    value = b.most_common(1)
    # print(value)
    if value[0][0] == 1:
        answer.append("T")
    else:
        answer.append("F")

    b = Counter(PJ)
    value = b.most_common(1)
    # print(value)
    if value[0][0] == 1:
        answer.append("P")
    else:
        answer.append("J")
    mbti = "".join(answer)
    # Classifying Personality's ==========================================>
    # print('===============================================================================>')

    ENFJ = {"type":"The Giver","statement":"They are extroverted, idealistic, charismatic, outspoken, \n highly principled and ethical, and usually know how to connect \n with others no matter their background or personality."}
    ISTJ = {"type":"The Inspector","statement":"They appear serious, formal, and proper. They also love \n traditions and old-school values that uphold patience, hard work,\n honor, and social and cultural responsibility. They are reserved, calm, quiet, and upright."}
    INFJ = {"type":"The Counselor","statement":"They have a different, and usually more profound, way of\n looking at the world. They have a substance and depth in the way \n they think, never taking anything at surface level or accepting things the way they are"}
    INTJ = {"type":"The Mastermind","statement":"They are usually self-sufficient and would rather work \n alone than in a group. Socializing drains an introvert’s energy, \n causing them to need to recharge."}
    ISTP = {"type":"The Craftsman","statement":" They are mysterious people who are usually very rational\n and logical, but also quite spontaneous and enthusiastic. Their \n personality traits are less easily recognizable than those of other \n types, and even people who know them well can’t always anticipate \n their reactions. Deep down, ISTPs are spontaneous, unpredictable individuals,\n but they hide those traits from the outside world, often very successfully."}
    ESFJ = {"type":"The Provider","statement":" They are social butterflies, and their need to interact\n with others and make people happy usually ends up making them popular.\n The ESFJ usually tends to be the cheerleader or sports hero in high school\n and college. Later on in life, they continue to revel in the \n spotlight, and are primarily focused on organizing social events for their families,\n friends and communities. "}
    INFP = {"type":"The Idealist","statement":" They prefer not to talk about themselves, especially in \n the first encounter with a new person. They like spending time alone in \n quiet places where they can make sense of what is happening around them. \n They love analyzing signs and symbols, and consider them to be \n metaphors that have deeper meanings related to life. "}
    ESFP = {"type":"The Performer","statement":" They have an Extraverted, Observant, Feeling and Perceiving \n personality, and are commonly seen as Entertainers. Born to be in \n front of others and to capture the stage, ESFPs love the spotlight. ESFPs \n are thoughtful explorers who love learning and sharing what they \n learn with others. ESFPs are “people people” with strong interpersonal skills."}
    ENFP = {"type":"The Champion","statement":" They have an Extraverted, Intuitive, Feeling and Perceiving \n personality. This personality type is highly individualistic and \n Champions strive toward creating their own methods, looks, actions, habits, \n and ideas — they do not like cookie cutter people and hate when they \n are forced to live inside a box. "}
    ESTP = {"type":"The Doer","statement":" They have an Extraverted, Sensing, Thinking, and Perceptive\n personality. ESTPs are governed by the need for social interaction,\n feelings and emotions, logical processes and reasoning, along with a need \n for freedom. "}
    ESTJ = {"type":"The Supervisor","statement":" They are organized, honest, dedicated, dignified, traditional,\n and are great believers of doing what they believe is right and \n socially acceptable. Though the paths towards “good” and “right” are difficult,\n they are glad to take their place as the leaders of the pack. \n They are the epitome of good citizenry. "}
    ENTJ = {"type":"The Commander","statement":" Their secondary mode of operation is internal, where intuition \n and reasoning take effect. ENTJs are natural born leaders among \n the 16 personality types and like being in charge. They live in a world of \n possibilities and they often see challenges and obstacles as great \n opportunities to push themselves."}
    INTP = {"type":"The Thinker","statement":" They are well known for their brilliant theories and unrelenting\n logic, which makes sense since they are arguably the most logical \n minded of all the personality types. They love patterns, have a keen eye \n for picking up on discrepancies, and a good ability to read people, \n making it a bad idea to lie to an INTP."}
    ISFJ = {"type":"The Nurturer","statement":" They are philanthropists and they are always ready to give back \n and return generosity with even more generosity. The people \n and things they believe in will be upheld and supported with enthusiasm\n  and unselfishness."}
    ENTP = {"type":"The Visionary","statement":" Those with the ENTP personality are some of the rarest in the world,\n  which is completely understandable. Although they are \n extroverts, they don’t enjoy small talk and may not thrive in many social \n situations, especially those that involve people who are too different\n from the ENTP. ENTPs are intelligent and knowledgeable need to \n be constantly mentally stimulated. "}
    ELSE = {"type":"The Composer","statement":" They are introverts that do not seem like introverts. It is \n because even if they have difficulties connecting to other people\n at first, they become warm, approachable, and friendly eventually. They \n are fun to be with and very spontaneous, which makes them the perfect \n friend to tag along in whatever activity, regardless if planned \n or unplanned. ISFPs want to live their life to the fullest and embrace the\n present, so they make sure they are always out to explore new things and \n discover new experiences."}
    output = {"ENFJ":ENFJ,"ISTJ":ISTJ,"INFJ":INFJ,"INTJ":INTJ,
    "ISTP":ISTP,"ESFJ":ESFJ,"INFP":INFP,"ESFP":ESFP,"ENFP":ENFP,
    "ESTP":ESTP,"ESTJ":ESTP,"ENTJ":ENTJ,"INTP":INTP,"ISFJ":ISFJ,"ENTP":ENTP}

    # print('User Name - ' + username + '\n')

    # print('Type of Personality \n')

    # if mbti == 'ENFJ':
    #     str1 = '" The Giver "'
    #     print(mbti + ' - ' + str1)
    #     print("\nThe Giver :-")
    #     print(" They are extroverted, idealistic, charismatic, outspoken, \n highly principled and ethical, and usually know how to connect \n with others no matter their background or personality. ")

    # elif mbti == 'ISTJ':
    #     str1 = '" The Inspector "'
    #     print(mbti + ' - ' + str1)
    #     print("\nThe Inspector :-")
    #     print(" They appear serious, formal, and proper. They also love \n traditions and old-school values that uphold patience, hard work,\n honor, and social and cultural responsibility. They are reserved, calm, quiet, and upright. ")

    # elif mbti == 'INFJ':
    #     str1 = '" The Counselor "'
    #     print(mbti + ' - ' + str1)
    #     print("\nThe Counselor :-")
    #     print(" They have a different, and usually more profound, way of\n looking at the world. They have a substance and depth in the way \n they think, never taking anything at surface level or accepting things the way they are")

    # elif mbti == 'INTJ':
    #     str1 = '" The Mastermind "'
    #     print(mbti + ' - ' + str1)
    #     print("\nThe Mastermind :-")
    #     print(" They are usually self-sufficient and would rather work \n alone than in a group. Socializing drains an introvert’s energy, \n causing them to need to recharge.")

    # elif mbti == 'ISTP':
    #     str1 = '" The Craftsman "'
    #     print(mbti + ' - ' + str1)
    #     print("\nThe Craftsman :-")
    #     print(" They are mysterious people who are usually very rational\n and logical, but also quite spontaneous and enthusiastic. Their \n personality traits are less easily recognizable than those of other \n types, and even people who know them well can’t always anticipate \n their reactions. Deep down, ISTPs are spontaneous, unpredictable individuals,\n but they hide those traits from the outside world, often very successfully.")

    # elif mbti == 'ESFJ':
    #     str1 = '" The Provider "'
    #     print(mbti + ' - ' + str1)
    #     print("\nThe Provider :-")
    #     print(" They are social butterflies, and their need to interact\n with others and make people happy usually ends up making them popular.\n The ESFJ usually tends to be the cheerleader or sports hero in high school\n and college. Later on in life, they continue to revel in the \n spotlight, and are primarily focused on organizing social events for their families,\n friends and communities. ")

    # elif mbti == 'INFP':
    #     str1 = '" The Idealist "'
    #     print(mbti + ' - ' + str1)
    #     print("\nThe Idealist :-")
    #     print(" They prefer not to talk about themselves, especially in \n the first encounter with a new person. They like spending time alone in \n quiet places where they can make sense of what is happening around them. \n They love analyzing signs and symbols, and consider them to be \n metaphors that have deeper meanings related to life. ")

    # elif mbti == 'ESFP':
    #     str1 = '" The Performer "'
    #     print(mbti + ' - ' + str1)
    #     print("\nThe Performer :-")
    #     print(" They have an Extraverted, Observant, Feeling and Perceiving \n personality, and are commonly seen as Entertainers. Born to be in \n front of others and to capture the stage, ESFPs love the spotlight. ESFPs \n are thoughtful explorers who love learning and sharing what they \n learn with others. ESFPs are “people people” with strong interpersonal skills.")

    # elif mbti == 'ENFP':
    #     str1 = '" The Champion "'
    #     print(mbti + ' - ' + str1)
    #     print("\nThe Champion :-")
    #     print(" They have an Extraverted, Intuitive, Feeling and Perceiving \n personality. This personality type is highly individualistic and \n Champions strive toward creating their own methods, looks, actions, habits, \n and ideas — they do not like cookie cutter people and hate when they \n are forced to live inside a box. ")

    # elif mbti == 'ESTP':
    #     str1 = '" The Doer "'
    #     print(mbti + ' - ' + str1)
    #     print("\nThe Doer :-")
    #     print(" They have an Extraverted, Sensing, Thinking, and Perceptive\n personality. ESTPs are governed by the need for social interaction,\n feelings and emotions, logical processes and reasoning, along with a need \n for freedom. ")

    # elif mbti == 'ESTJ':
    #     str1 = '" The Supervisor "'
    #     print(mbti + ' - ' + str1)
    #     print("\nThe Supervisor :-")
    #     print(" They are organized, honest, dedicated, dignified, traditional,\n and are great believers of doing what they believe is right and \n socially acceptable. Though the paths towards “good” and “right” are difficult,\n they are glad to take their place as the leaders of the pack. \n They are the epitome of good citizenry. ")

    # elif mbti == 'ENTJ':
    #     str1 = '" The Commander "'
    #     print(mbti + ' - ' + str1)
    #     print("\nThe Commander :-")
    #     print(" Their secondary mode of operation is internal, where intuition \n and reasoning take effect. ENTJs are natural born leaders among \n the 16 personality types and like being in charge. They live in a world of \n possibilities and they often see challenges and obstacles as great \n opportunities to push themselves.")

    # elif mbti == 'INTP':
    #     str1 = '" The Thinker "'
    #     print(mbti + ' - ' + str1)
    #     print("\nThe Thinker :-")
    #     print(" They are well known for their brilliant theories and unrelenting\n logic, which makes sense since they are arguably the most logical \n minded of all the personality types. They love patterns, have a keen eye \n for picking up on discrepancies, and a good ability to read people, \n making it a bad idea to lie to an INTP.")

    # elif mbti == 'ISFJ':
    #     str1 = '" The Nurturer "'
    #     print(mbti + ' - ' + str1)
    #     print("\nThe Nurturer :-")
    #     print(" They are philanthropists and they are always ready to give back \n and return generosity with even more generosity. The people \n and things they believe in will be upheld and supported with enthusiasm\n  and unselfishness.")

    # elif mbti == 'ENTP':
    #     str1 = '" The Visionary "'
    #     print(mbti + ' - ' + str1)
    #     print("\nThe Visionary :-")
    #     print(" Those with the ENTP personality are some of the rarest in the world,\n  which is completely understandable. Although they are \n extroverts, they don’t enjoy small talk and may not thrive in many social \n situations, especially those that involve people who are too different\n from the ENTP. ENTPs are intelligent and knowledgeable need to \n be constantly mentally stimulated. ")

    # else:
    #     str1 = '" The Composer "'
    #     print(mbti + ' - ' + str1)
    #     print("\nThe Composer :-")
    #     print(" They are introverts that do not seem like introverts. It is \n because even if they have difficulties connecting to other people\n at first, they become warm, approachable, and friendly eventually. They \n are fun to be with and very spontaneous, which makes them the perfect \n friend to tag along in whatever activity, regardless if planned \n or unplanned. ISFPs want to live their life to the fullest and embrace the\n present, so they make sure they are always out to explore new things and \n discover new experiences.")


    return username,mbti,output.get(mbti,ELSE)

if __name__ == "__main__":
    print('===============================================================================>')
    username,mbti,personality = classify_user("amanthe001")
    print('User Name - ' + username + '\n')
    print('Type of Personality \n')
    print(mbti + ' - ' + personality["type"])
    print(personality["type"],":-")
    print(personality["statement"])
    print('===============================================================================>')

