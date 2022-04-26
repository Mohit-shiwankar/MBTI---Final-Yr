import csv #importing csv files and performing operations on csv
import array
import pandas # Pandas provide an easy way to create, manipulate, and wrangle the data.
import pickle # for serializing and de-serializing a Python object structure.
import os #named PATH by using which we can perform many more functions.
import sys #
import numpy as np #to handle array and perform numerical operation on arrray
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer #count term frequency–inverse document frequency
#tokenizing strings and giving an integer id for each possible token, for instance by using white-spaces and punctuation as token separators.counting the occurrences of tokens in each document.normalizing and weighting with diminishing importance tokens that occur in the majority of samples / documents.
from sklearn.naive_bayes import GaussianNB #Gaussian Naive Bayes is the easiest and rapid classification method available.
from sklearn import svm
csvFile=open('C:/Users/Abhi/Desktop/Dataset/newfrequency300.csv', 'rt')
csvReader=csv.reader(csvFile)
mydict={row[1]: int(row[0]) for row in csvReader}

y=[]
with open ('C:/Users/Abhi/Desktop/Dataset/PJFinaltest.csv', 'rt') as f:
	reader=csv.reader(f)
	corpus=[rows[0] for rows in reader]

with open ('C:/Users/Abhi/Desktop/Dataset/PJFinaltest.csv', 'rt') as f:
	csvReader1=csv.reader(f)
	for rows in csvReader1:
		y.append([int(rows[1])])
vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1,lowercase=False)
x=vectorizer.fit_transform(corpus).toarray()
result=np.append(x,y,axis=1)
X=pandas.DataFrame(result)
model=GaussianNB()
train = X.sample(frac=0.8, random_state=1)
test=X.drop(train.index)
y_train=train[301]
y_test=test[301]
#print(train.shape)
#print(test.shape)
xtrain=train.drop(301,axis=1)
xtest=test.drop(301,axis=1)
model.fit(xtrain,y_train)
pickle.dump(model, open('C:/Users/Abhi/Desktop/Twitter/BNPJFinal.sav', 'wb')) #To save your model in dump is used where 'wb' means write binary. pickle.dump (model, open (filename, 'wb')) #Saving the model To load the saved model wherever need load is used where 'rb' means read binary. model = pickle.load (open (filename, 'rb')) #To load saved model from local directory
del result

y=[]
with open ('C:/Users/Abhi/Desktop/Dataset/IEFinaltest.csv', 'rt') as f:
	reader=csv.reader(f)
	corpus=[rows[0] for rows in reader]

with open ('C:/Users/Abhi/Desktop/Dataset/IEFinaltest.csv', 'rt') as f:
	csvReader1=csv.reader(f)
	for rows in csvReader1:
		y.append([int(rows[1])])
vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(corpus).toarray()
result=np.append(x,y,axis=1)
X=pandas.DataFrame(result)
model=GaussianNB()
train = X.sample(frac=0.8, random_state=1)
test=X.drop(train.index)
y_train=train[301]
y_test=test[301]
#print(train.shape)
#print(test.shape)
xtrain=train.drop(301,axis=1)
xtest=test.drop(301,axis=1)
model.fit(xtrain,y_train)
pickle.dump(model, open('C:/Users/Abhi/Desktop/Twitter/BNIEFinal.sav', 'wb')) #binary file formats may significantly improve the throughput of your import pipeline, helping reduce model training time.
del result

y=[]
with open ('C:/Users/Abhi/Desktop/Dataset/TFFinaltest.csv', 'rt') as f:
	reader=csv.reader(f)
	corpus=[rows[0] for rows in reader]

with open ('C:/Users/Abhi/Desktop/Dataset/TFFinaltest.csv', 'rt') as f:
	csvReader1=csv.reader(f)
	for rows in csvReader1:
		y.append([int(rows[1])])
vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(corpus).toarray()
result=np.append(x,y,axis=1)
X=pandas.DataFrame(result)
model=GaussianNB()
train = X.sample(frac=0.8, random_state=1)
test=X.drop(train.index)
y_train=train[301]
y_test=test[301]
#print(train.shape)
#print(test.shape)
xtrain=train.drop(301,axis=1)
xtest=test.drop(301,axis=1)
model.fit(xtrain,y_train)
pickle.dump(model, open('C:/Users/Abhi/Desktop/Twitter/BNTFFinal.sav', 'wb'))#The python dump function is used by importing packages like json and pickle in python and the basic syntax for both the functions is,
del result

y=[]
with open ('C:/Users/Abhi/Desktop/Dataset/SNFinaltest.csv', 'rt') as f:
	reader=csv.reader(f)
	corpus=[rows[0] for rows in reader]

with open ('C:/Users/Abhi/Desktop/Dataset/SNFinaltest.csv', 'rt') as f:
	csvReader1=csv.reader(f)
	for rows in csvReader1:
		y.append([int(rows[1])])
vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(corpus).toarray() #joins these two steps and is used for the initial fitting of parameters on the training set
result=np.append(x,y,axis=1)
X=pandas.DataFrame(result)
model=GaussianNB()
train = X.sample(frac=0.8, random_state=1)
test=X.drop(train.index)
y_train=train[301]
y_test=test[301]
#print(train.shape)
#print(test.shape)
xtrain=train.drop(301,axis=1)
xtest=test.drop(301,axis=1)
model.fit(xtrain,y_train)
pickle.dump(model, open('C:/Users/Abhi/Desktop/Twitter/BNSNFinal.sav', 'wb'))





from nltk.corpus import stopwords # NLTK is a toolkit build for working with NLP in Python. It provides us various text processing libraries with a lot of test datasets.A stop word is a commonly used word (such as 'the') that a search engine has been programmed to ignore.
from nltk.tokenize import word_tokenize #Tokenization is the process by which big quantity of text is divided into smaller parts called tokens
from nltk.stem import * #Stemming is one of the most used techniques used for text normalization.
from nltk.stem.snowball import SnowballStemmer
import tweepy #An easy-to-use Python library for accessing the Twitter API.
import sys
import os #changing the directory or path of the files
import nltk # To work efficiently with the natural language.
import re #regular expression
import numpy as np # To perform nummerical operation in dataset.
import string
from unidecode import unidecode #takes Unicode data and tries to represent it in ASCII characters
import csv
from itertools import islice
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import nltk
nltk.download('stopwords') #to remove stopwords from the text
import nltk
nltk.download('punkt') # to remove puntuations marks from text
ckey='QWUohV4OV7QIwxN7mVvqh93wo'
csecret='ex1GL5B8jHYzCSKUmTofMacXrlA6hh8tgTgbRAI1IafMQJJygD'
atoken='1180377247791927298-kWpV5SsxQkCyqIWdBOcUdsJ66To504'
asecret='jEdJSlU4XRJ0lbTYddlCfjKm15PSQAUHXfwtQ9G5L5zTL'
auth=tweepy.OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
api=tweepy.API(auth)

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
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=True):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def preproc(s):
	#s=emoji_pattern.sub(r'', s) # no emoji
	s= unidecode(s)
	POSTagger=preprocess(s) #Part-of-speech tagging is one of the most important text analysis tasks used to classify words into their part-of-speech and label them according the tagset which is a collection of tags used for the pos tagging. Part-of-speech tagging also known as word classes or lexical categories.
	#print(POSTagger)

	tweet=' '.join(POSTagger)
	stop_words = set(stopwords.words('english'))
	word_tokens = word_tokenize(tweet)
	#filtered_sentence = [w for w in word_tokens if not w in stop_words]
	filtered_sentence = []
	for w in POSTagger:
	    if w not in stop_words:
	        filtered_sentence.append(w)
	#print(word_tokens)
	#print(filtered_sentence)
	stemmed_sentence=[]
	stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
	for w in filtered_sentence:
		stemmed_sentence.append(stemmer2.stem(w))
	#print(stemmed_sentence)

	temp = ' '.join(c for c in stemmed_sentence if c not in string.punctuation)
	preProcessed=temp.split(" ")
	final=[]
	for i in preProcessed:
		if i not in final:
			if i.isdigit():
				pass
			else:
				if 'http' not in i:
					final.append(i)
	temp1=' '.join(c for c in final)
	#print(preProcessed)
	return temp1

def getTweets(user):
	csvFile = open('C:/Users/Abhi/Desktop/Twitter/user.csv', 'a', newline='')
	csvWriter = csv.writer(csvFile)
	try:
		for i in range(0,4):
			tweets=api.user_timeline(screen_name = user, count =10, include_rts=True, page=i)
			for status in tweets:
				tw=preproc(status.text)
				if tw.find(" ") == -1:
					tw="blank"
				csvWriter.writerow([tw])
	except tweepy.TweepError:
		print("Failed to run the command on that user, Skipping...")
	csvFile.close()
	#stopwords.words('english')

username=input("Please Enter Twitter Account handle: ")
getTweets(username)
with open('C:/Users/Abhi/Desktop/Twitter/user.csv','rt') as f:
	csvReader=csv.reader(f)
	tweetList=[rows[0] for rows in csvReader]
with open('C:/Users/Abhi/Desktop/Dataset/newfrequency300.csv','rt') as f:
	csvReader=csv.reader(f)
	mydict={rows[1]: int(rows[0]) for rows in csvReader}

vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(tweetList).toarray()
df=pd.DataFrame(x)


model_IE = pickle.load(open('C:/Users/Abhi/Desktop/Twitter/BNIEFinal.sav', 'rb'))
model_SN = pickle.load(open('C:/Users/Abhi/Desktop/Twitter/BNSNFinal.sav', 'rb'))
model_TF = pickle.load(open('C:/Users/Abhi/Desktop/Twitter/BNTFFinal.sav', 'rb'))
model_PJ = pickle.load(open('C:/Users/Abhi/Desktop/Twitter/BNPJFinal.sav', 'rb'))

answer=[]
IE=model_IE.predict(df)
SN=model_SN.predict(df)
TF=model_TF.predict(df)
PJ=model_PJ.predict(df)


b = Counter(IE)
value=b.most_common(1)
#print(value)
if value[0][0] == 1.0:
	answer.append("I")
else:
	answer.append("E")

b = Counter(SN)
value=b.most_common(1)
#print(value)
if value[0][0] == 1.0:
	answer.append("S")
else:
	answer.append("N")

b = Counter(TF)
value=b.most_common(1)
#print(value)
if value[0][0] == 1:
	answer.append("T")
else:
	answer.append("F")

b = Counter(PJ)
value=b.most_common(1)
#print(value)
if value[0][0] == 1:
	answer.append("P")
else:
	answer.append("J")
mbti="".join(answer)
print(mbti)
if (mbti=="INFP"):
	print("I=Intraversion")
	print("N=Intuition")
	print("F=Feeling")
	print("P=Percieiving")
elif(mbti=="INFJ"):
	print("I=Intraversion")
	print("N=Intuition")
	print("F=Feeling")
	print("J=Judging")
elif(mbti=="ENFP"):
	print("E=Entraversion")
	print("N=Intuition")
	print("F=Feeling")
	print("P=Percieiving")
elif(mbti=="INTJ"):
	print("I=Intraversion")
	print("N=Intuition")
	print("F=Feeling")
	print("J=Judging")
elif(mbti=="INTP"):
	print("I=Intraversion")
	print("N=Intuition")
	print("T=Thinking")
	print("P=Percieiving")
elif(mbti=="ISFJ"):
	print("I=Intraversion")
	print("S=Sensing")
	print("F=Feeling")
	print("J=Judging")
elif(mbti=="ENFJ"):
	print("E=Entraversion")
	print("N=Intuition")
	print("F=Feeling")
	print("J=Judging")
elif(mbti=="ENTJ"):
	print("E=Entraversion")
	print("N=Intuition")
	print("T=Thinking")
	print("J=Judging")
elif(mbti=="ISTJ"):
	print("I=Intraversion")
	print("S=Sensing")
	print("T=Thinking")
	print("J=Judging")
elif(mbti=="ESTJ"):
	print("E=Entraversion")
	print("S=Sensing")
	print("T=Thinking")
	print("J=Judging")
elif(mbti=="ISFP"):
	print("I=Intraversion")
	print("S=Sensing")
	print("F=Feeling")
	print("P=Percieiving")
elif(mbti=="ISTP"):
	print("I=Intraversion")
	print("S=Sensing")
	print("T=Thinking")
	print("P=Percieiving")
elif(mbti=="ESFJ"):
	print("E=Entraversion")
	print("S=Sensing")
	print("F=Feeling")
	print("J=Judging")
elif(mbti=="ESTP"):
	print("E=Entraversion")
	print("S=Sensing")
	print("T=Thinking")
	print("P=Percieiving")
elif(mbti=="ESFP"):
	print("E=Entraversion")
	print("S=Sensing")
	print("F=Feeling")
	print("P=Percieiving")
print(" ")

import os
if os.path.exists("C:/Users/Abhi/Desktop/Twitter/BNIEFinal.sav"):
  os.remove("C:/Users/Abhi/Desktop/Twitter/BNIEFinal.sav")
if os.path.exists("C:/Users/Abhi/Desktop/Twitter/BNSNFinal.sav"):
  os.remove("C:/Users/Abhi/Desktop/Twitter/BNSNFinal.sav")
if os.path.exists("C:/Users/Abhi/Desktop/Twitter/BNTFFinal.sav"):
  os.remove("C:/Users/Abhi/Desktop/Twitter/BNTFFinal.sav")
if os.path.exists("C:/Users/Abhi/Desktop/Twitter/BNPJFinal.sav"):
  os.remove("C:/Users/Abhi/Desktop/Twitter/BNPJFinal.sav")
#if os.path.exists("C:/Users/Abhi/Desktop/Twitter/user.csv"):
# os.remove("C:/Users/Abhi/Desktop/Twitter/user.csv")
