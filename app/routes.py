from app import app
from flask import Flask, render_template, request
from flask_cors import cross_origin
from app.utils import get_response, make_response, predict_class,modal,intents,no_accent_vietnamese
from datetime import datetime

@app.route('/')
@cross_origin()
def index():
	data = dict(data='this is botchat api')
	return make_response(data)

@app.route("/get" , methods=['GET', 'POST'])
@cross_origin()
def get_bot_response():
    sentence = request.get_json(force=True)
    sentence = no_accent_vietnamese(sentence['data'])
    results = predict_class(sentence, modal)
    bot = ""
    if (len(results) > 0):
        if (results[0][0] == 'time'):
            bot = "Bây giờ là: " + datetime.now().strftime("%H:%M")
        elif (results[0][0] == 'date'):
            bot = "hôm nay là ngày: " + datetime.now().strftime("%d/%m/%Y")
        else:
            bot = get_response(results, intents)
    else:
        bot = "Nói gì hỏng hiểu :<"
    return make_response({"bot": bot})
