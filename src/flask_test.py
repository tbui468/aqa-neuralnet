import os
import json
from flask import Flask
from flask import request

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab



#make vocab dictionary with word (key) and word-count (value)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')
counter = Counter()
for (label, line) in train_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter, min_freq=1)
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


vocab_size = len(vocab)
emsize = 64
num_class = 4
model = TextClassificationModel(vocab_size, emsize, num_class)
model.load_state_dict(torch.load('text_classification_weights.pth'))
model.eval()

topic = {
            0: "Sports",
            1: "Business",
            2: "Science/Tech"
        }

print("loaded model!")


app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello flasky boy'


@app.route('/predict', methods=['POST'])
def pred():
    if request.method == 'POST':
        text = request.get_json()
        processed_text = torch.tensor(text_pipeline(text["text"]), dtype=torch.int64)

        with torch.no_grad():
            pred = model(processed_text, torch.tensor([0]))
            #print(pred.numpy()[0]) #this returns a 1-d array that we can now use to extract topic, just slice the tensor to remove world news
            return { "topic": topic[pred[:, 1:].argmax(1).item()] } #removing first index (belonging to world news)


#FLASK_ENV=development FLASK_APP=flask_test.py flask run
