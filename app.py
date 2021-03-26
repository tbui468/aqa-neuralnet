import os
import json
from flask import Flask
from flask import request
from flask import make_response


import torch
from torch.utils.data import DataLoader
from torchtext.datasets import YahooAnswers



#make vocab dictionary with word (key) and word-count (value)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


topic = {
            0: "Society and Culture",
            1: "Science and Mathematics",
            2: "Health",
            3: "Education and Reference",
            4: "Computers and Internet",
            5: "Sports",
            6: "Business and Finance",
            7: "Entertainment and Music",
            8: "Family and Relationships",
            9: "Politics and Government"
        }

import network


vocab = network.read_vocab('yahoo_vocab.txt')

model = network.QAClassificationModel(len(vocab), 64, len(topic))
model.load_state_dict(torch.load('yahoo_model_weights.pth'))
model.eval()


print("loaded model!")


app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello flasky boy'


@app.route('/predict', methods=['POST'])
def pred():
    if request.method == 'POST':
        text = request.get_json()
        #processed_text = torch.tensor(text_pipeline(text["text"]), dtype=torch.int64)
        with torch.no_grad():
            pred = network.predict_text(model, text["text"], vocab)
            return { "topic": topic[pred.argmax(1).item()] }
            #pred = model(processed_text, torch.tensor([0]))
            #return { "topic": topic[pred[:, 1:].argmax(1).item()] } #removing first index (belonging to world news)


#FLASK_ENV=development FLASK_APP=flask_test.py flask run
