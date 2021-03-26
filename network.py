from collections import Counter

import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab


class CollateManager(object):
    def __init__(self, vocab): #for simplicity, this should just take a dictionary object
        self.vocab = vocab
        self.tokenizer = get_tokenizer('basic_english')

    def __call__(self, batch):
        labels_list, tokens_list, offsets_list = [], [], [0] #will push in length after each text's tokens are appended, then slice off last value at end
        for label, text in batch:
            labels_list.append(int(label) - 1)
            tokens = [self.vocab[token] for token in self.tokenizer(text)]
            tokens_list.extend(tokens)
            offsets_list.append(len(tokens))

        labels_list = torch.tensor(labels_list)
        tokens_list = torch.tensor(tokens_list)
        offsets_list = torch.tensor(offsets_list[:-1]).cumsum(dim=0)

        return labels_list, tokens_list, offsets_list




class QAClassificationModel(nn.Module):
    def __init__(self, vocab_count, embed_dim, class_count):
        super(QAClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_count, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, class_count)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, tokens_list, offsets_list):
        embedded = self.embedding(tokens_list, offsets_list)
        return self.fc(embedded)


def create_vocab(train_set):
    counter = Counter()
    tokenizer = get_tokenizer('basic_english')
    for label, text in train_set:
        counter.update(tokenizer(text))
    return Vocab(counter, min_freq=1)

def train(model, data_loader):
    model.train()
    criterion = torch.nn.CrossEntropyLoss() #read about cross entropy loss, and how it's different from just finding difference between labels
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    for idx, (labels_list, tokens_list, offsets_list) in enumerate(data_loader):
        optimizer.zero_grad()
        pred = model(tokens_list, offsets_list)
        loss = criterion(pred, labels_list)
        loss.backward()
        optimizer.step()


def evaluate(model, data_loader):
    model.eval()
    correct = 0.0
    total = 0.0
    for idx, (labels_list, tokens_list, offsets_list) in enumerate(data_loader):
        output = model(tokens_list, offsets_list)
        pred = output.argmax(1)
        total += len(labels_list)
        for i in range(0, len(pred)):
            if(pred[i] == labels_list[i]): correct += 1

    print("Accuracy: ", correct/total * 100.0, "%")


#saving token and index (tab separated)
def save_vocab(vocab, path):
    with open(path, 'w+') as f:
        for token, index in vocab.stoi.items():
            f.write(f'{index}\t{token}\n')


#reading in file 
def read_vocab(path):
    vocab = dict()
    with open(path, 'r') as f:
        for line in f:
            index, token = line.split('\t')
            vocab[token[:-1]] = int(index) #[:-1] is to remove \n character at end of each word
    return vocab


def predict_text(model, text, vocab):
    unk_idx = 0
    tokenizer = get_tokenizer('basic_english')
    tokens = tokenizer(text)
    indexes = [vocab.get(t, unk_idx) for t in tokens]
    #tensor = torch.LongTensor(indexes).unsqueeze(1)
    output = model(torch.tensor(indexes), torch.tensor([0]))
    return output


