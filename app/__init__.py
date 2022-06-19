from flask import Flask, request, jsonify
import os
from flask_cors import CORS

# app = Flask(__name__)
app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "https://about-me-goiliace.vercel.app/"}}) # change from '*' to this route 
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from app.routes import *
from app.utils import *
