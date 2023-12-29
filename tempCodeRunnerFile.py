import speech_recognition as sr
import pyaudio
import pyttsx3

#init_rec = sr.Recognizer()
#print("Let's speak!!")
#with sr.Microphone() as source:
    #audio_data = init_rec.record(source, duration=10)
    #print("Recognizing your text.............")
    #det = init_rec.recognize_google(audio_data)
    #print(det)

    # writing the detected audio to file
det=input("Enter the message: ")
print("Loading the Detection model.....")
print("Gathering the sources.....")
# Opening a file
file1 = open('myfile.txt', 'w')
L = [det]

    # Writing multiple strings
    # at a time
file1.writelines(L)
file1.close()

    # written to file or not
file1 = open('myfile.txt', 'r')
#print(file1.read())
file1.close()

#text to speech conversion
#from gtts import gTTS  
#import os
  
# The text that you want to convert to audio
#mytext = input("Enter your text here : ")

# Language in which you want to convert
#language = 'en'
  
#myobj = gTTS(text="Hello, You just spoke       " + det, lang=language, slow=False)
# Saving the converted audio in a mp3 file named
# welcome 
#myobj.save("welcome.mp3")
  
# Playing the converted file
#os.system("welcome.mp3")
#engine = pyttsx3.init('sapi5')
#voice= engine.setProperty('voice')
#engine.setProperty('voice', voice[1].id)
#engine.setProperty('rate', 150)
#engine.say(" Hello user You just spoke     " + det)
#engine.runAndWait()

from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import nltk
nltk.download('stopwords')

data = pd.read_csv("twitter.csv")
#print(data.head())

data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
#print(data.head())

data = data[["tweet", "labels"]]
#print(data.head())

import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)
#print(data.head())

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

def hate_speech_detection():
    data = cv.transform([det]).toarray()
    a = clf.predict(data)
    #print(a)
    txt_speech = pyttsx3.init()
    txt_speech.setProperty('rate', 150)
    txt = a
    print(txt+" detected")
    txt_speech.say(txt + '   detected  ')
    txt_speech.runAndWait()
hate_speech_detection()
