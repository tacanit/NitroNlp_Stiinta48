from curses import window
import re
import string
import numpy as np
import nltk
import pandas as pd
import datasets
import spacy
import torchvision
import torch
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from datasets import load_dataset
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import MWETokenizer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from spacy.lang.ro.examples import sentences
from tqdm import trange
from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('averaged_perceptron_tagger')
nlp = spacy.load("en_core_web_sm")

from datasets import load_dataset
ronec = load_dataset("ronec")
dataset = [
    ' '.join([token for token in sample["tokens"] if token not in string.punctuation]) \
    for sample in ronec["train"]
]

alphabet_size = 32
window_size = 10
complete_text = ' '.join(dataset)
text_size = len(complete_text)
characters, counts = np.unique(list(complete_text.lower()), return_counts=True)
char_count_pairs = list(zip(characters, counts))
char_count_pairs.sort(key=lambda item: item[1], reverse=True)
alphabet_pairs = char_count_pairs[:alphabet_size]
alphabet_chars, alphabet_counts = zip(*alphabet_pairs)
plt.figure(figsize=(18, 7))
plt.barh(alphabet_chars, alphabet_counts)
plt.tight_layout()
alphabet_map = {char: idx + 2 for idx, char in enumerate(characters)}
alphabet_map['<unk>'] = 0
letter_ids = np.zeros(text_size, np.int16)

for i in trange(text_size):
    char = complete_text[i].lower()
    letter_ids[i] = alphabet_map.get(char, 0)

windows = np.zeros([text_size, window_size], np.int16)
labels = np.zeros(text_size, np.uint8)

full_data = []

for i in trange(text_size - window.size):
    windows[i] - letter_ids[i: i + window_size]
    labels[i] = complete_text[i].isupper()
    full_data.append({
        "windows": windows,
        "labels": labels
    })

train_test_split