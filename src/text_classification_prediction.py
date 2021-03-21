import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#make vocab dictionary with word (key) and word-count (value)
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
print("loaded model!")

ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

text2 = "A new study of the U.K. and South Africa variants of SARS-CoV-2 predicts that current vaccines and certain monoclonal \
        antibodies may be less effective at neutralizing these variants and that the new variants raise the specter that reinfections could be more likely.\
        The study's predictions are now being borne out with the first reported results of the Novavax vaccine, says the \
        study's lead author David Ho, MD. The company reported on Jan. 28 that the vaccine was nearly 90% effective in the \
        company's U.K. trial, but only 49.4% effective in its South Africa trial, where most cases of COVID-19 are caused by the B.1.351 variant."

text3 = "What is a key perfomance indicator? \
A performance indicator or key performance indicator (KPI) is a type of performance measurement. KPIs evaluate the \
success of an organization or of a particular activity (such as projects, programs, products and other initiatives) in which it engages. \
Often success is simply the repeated, periodic achievement of some levels of operational goal (e.g. zero defects, 10/10 \
customer satisfaction), and sometimes success is defined in terms of making progress toward strategic goals. \
Accordingly, choosing the right KPIs relies upon a good understanding of what is important to the organization.  \
What is deemed important often depends on the department measuring the performance – e.g. the KPIs useful to finance will \
differ from the KPIs assigned to sales. \
Since there is a need to understand well what is important, various techniques to assess the present state of the business,  \
and its key activities, are associated with the selection of performance indicators. These assessments often lead to the  \
identification of potential improvements, so performance indicators are routinely associated with 'performance improvement'  \
initiatives. A very common way to choose KPIs is to apply a management framework such as the balanced scorecard. "

processed_text = torch.tensor(text_pipeline(ex_text_str), dtype=torch.int64)

with torch.no_grad():
    pred = model(processed_text, torch.tensor([0]))
    print(pred)


print('prediction done!')
