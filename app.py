import random
import numpy as np
import json
import pickle
import nltk
from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from flask import jsonify
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
# nltk.data.path.append('./nltk_data/')
app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "https://fuzzy-c-mean-fe.vercel.app"}}) # change from '*' to this route
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from flask import Flask, render_template, request
from flask_cors import cross_origin
from datetime import datetime
nltk.download()
@app.route('/')
@cross_origin()
def index():
	data = dict(data='this is botchat api')
	return make_response(data)

@app.route("/get")
@cross_origin()
def get_bot_response():
    sentence =request.args.get('mess')
    results = predict_class(sentence, modal)
    bot = ""
    if (len(results) > 0):
        if (results[0][0] == 'time'):
            # print('Bot: ' + datetime.now().strftime("%H:%M:%S"))
            bot = datetime.now().strftime("%H:%M:%S")
        elif (results[0][0] == 'date'):
            # print('Bot: ' + datetime.now().strftime("%d/%m/%Y"))
            bot = datetime.now().strftime("%d/%m/%Y")
        else:
            # print('Bot: ', get_response(results, intents))
            bot = get_response(results, intents)
    else:
        # print("Bot: I'm not sure what you mean.")
        bot = "Hưng chưa lập trình cho tình huống này :("
    return make_response({"bot": bot})

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json', encoding='utf8').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
modal = load_model('chatbot_model.h5')

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

if __name__ == '__main__':
    app.run(debug=True)
