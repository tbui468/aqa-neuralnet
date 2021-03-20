import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello flasky'


#FLASK_ENV=development FLASK_APP=flask_test.py flask run
