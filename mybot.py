import random
import json
import numpy as np
import tensorflow as tf
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

from keras.models import load_model

lemmatizer= WordNetLemmatizer()
with open("intents.json",'r') as file:
    intents= json.load(file)

with open("words.pkl", "rb") as file:
    words = pickle.load(file)
with open("classes.pkl", "rb") as file:
    classes = pickle.load(file)

model=load_model("chatbot_apoorva.keras")

def clean_up_sentence(sentence):
    sentence_words= nltk.word_tokenize(sentence)
    lem_words= [lemmatizer.lemmatize(word) for word in sentence_words] 
    return lem_words

def bag_of_words(sentence):
    sentence_words= clean_up_sentence(sentence)
    bag= [0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word==w:
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bow= bag_of_words(sentence)
    res=model.predict(np.array([bow]))
    res=res[0]

    ERROR_THRESHOLD=0.25
    results=[[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1],reverse=True)
    result_list=[]
    for r in results:
        result_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return result_list

def get_response(intents_list,intents_json):
    list_of_intents= intents_json['intents']
    tag=intents_list[0]['intent']
    for i in list_of_intents:
        if i['tag']==tag:
            result=random.choice(i['responses'])
            break

    return result

print("Our chatbot is running")

while True:
    user= input("You:")
    msg_class = predict_class(user)
    res=get_response(msg_class,intents)
    print(res)
