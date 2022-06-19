from app import app
from flask import Flask, render_template, request
from flask_cors import cross_origin
from app.utils import get_response, make_response, predict_class,modal,intents
from datetime import datetime
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
