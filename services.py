import torch
from models import CNNModel1,CNNModel2
from gensim.models import Word2Vec
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

from schemas import Review_out


nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()
device = torch.device('cpu')

w2v = Word2Vec.load('./w2vmodel.pth')
w2v.build_vocab([["UNK"]], update=True)
unk_vector = w2v.wv.vectors.mean(axis=0)
w2v.wv["UNK"] = unk_vector

model1 = CNNModel1(50)
model2 = CNNModel2(50)

model1.load_state_dict(torch.load('model1.pth',map_location=device))
model2.load_state_dict(torch.load('model2.pth',map_location=device))
word2idx = {word: idx for idx, word in enumerate(w2v.wv.index_to_key)}

def encode(sen):
    return np.array([word2idx[word] if word in word2idx else word2idx['UNK'] for word in sen.split()],dtype='int')
def clean_text(text):
    text = re.sub(r'<[^>]+>',' ',text, re.UNICODE)
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

def encode_text(text):
    x = text
    x = clean_text(x)
    x = encode(x).reshape(1,-1)
    x = torch.tensor(x,dtype=torch.long).to(device)
    return x


def get_review_out(text):
    x = encode_text(text)
    model1.eval()
    model2.eval()
    with torch.no_grad():
        sentiment = 'pos' if model1(x)>0.5 else 'neg'
        rate = torch.round(model2(x))
    return Review_out(text=text,sentiment=sentiment,rate=rate)