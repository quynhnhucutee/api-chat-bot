import random
import numpy as np
import json
import pickle
import nltk
from flask import jsonify
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('all')
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('data/intents.json', encoding='utf8').read())

words = pickle.load(open('data/words.pkl', 'rb'))
classes = pickle.load(open('data/classes.pkl', 'rb'))
modal = load_model('data/chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    bag = [0]*len(words)
    sentence_words = clean_up_sentence(sentence)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bag_of_words(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.97
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list
def make_response(data={}, status=200):
    '''
        - Make a resionable response with header
        - status default is 200 mean ok
    '''
    res = jsonify(data)
    # res.headers.add('Access-Control-Allow-Origin', 'http://127.0.0.1:5502')
    res.headers.add('Content-Type', 'application/json')
    res.headers.add('Accept', 'application/json')
    return res
def get_response(intent_list,intents_json):
    tag = intent_list[0][0]
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
