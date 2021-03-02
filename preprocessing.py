#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import spacy
import unidecode
from word2number import w2n
from pycontractions import Contractions
# import gensim.downloader as api
from gensim.models import KeyedVectors
from tqdm import tqdm

# nlp = spacy.load('en_core_web_md')
nlp = spacy.load('en_core_web_lg')

# Choose model accordingly for contractions function
# model = api.load("glove-twitter-25")
# model = api.load("glove-twitter-100")
# model = api.load("word2vec-google-news-300")

# Save model to disk
# model.save('word2vec-google-news-300')

# Load keyedvectors model
model = KeyedVectors.load('word2vec-google-news-300', mmap='r')

# Pass keyedvectors model in to Contractions
cont = Contractions(kv_model=model)

cont.load_models()


# In[2]:


# exclude words from spacy stopwords list
deselect_stop_words = ['no', 'not']
for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False

def remove_urls(text):
    text = text.replace(r'(https|http)?:\/(\w|\.|\/|\?|\=|\&|\%)*\b','')
    text = text.replace(r'www\.\S+\.com','')
    return text
    
def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())


def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text


def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = list(cont.expand_texts([text], precise=False))[0]
    return text


def text_preprocessing(text, accented_chars=True, contractions=True, 
                       convert_num=False, extra_whitespace=True, 
                       lemmatization=True, lowercase=True, punctuations=False,
                       remove_html=True, remove_num=False, special_chars=True, 
                       stop_words=False, urls=True):
    """preprocess text with default option set to true for all steps"""
    if urls == True: #remove urls
        text = remove_urls(text)
    if remove_html == True: #remove html tags
        text = strip_html_tags(text)
    if extra_whitespace == True: #remove extra whitespaces
        text = remove_whitespace(text)
    if accented_chars == True: #remove accented characters
        text = remove_accented_chars(text)
    if contractions == True: #expand contractions
        text = expand_contractions(text)
    if lowercase == True: #convert all characters to lowercase
        text = text.lower()

    doc = nlp(text) #tokenize text
    req_tag = ['NN', 'NNS', 'JJ', 'JJR', 'JJS'] #extract nouns from POS tagged text
    clean_text = []
    extracted_words = []
    
    for token in doc:
        flag = True
        edit = token.text
        # remove stop words
        if stop_words == True and token.is_stop and token.pos_ != 'NUM':
            flag = False
        # remove urls
        if urls == True and token.like_url and flag == True:
            flag = False
        # remove punctuations
        if punctuations == True and token.pos_ == 'PUNCT' and flag == True:
            flag = False
        # remove special characters
        if special_chars == True and token.pos_ == 'SYM' and flag == True:
            flag = False
        # remove numbers
        if remove_num == True and (token.pos_ == 'NUM' or token.text.isnumeric()) and flag == True:
            flag = False
        # convert number words to numeric numbers
        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            edit = w2n.word_to_num(token.text)
        # convert tokens to base form
        if lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
        # append tokens edited and not removed to list
        if edit != "" and flag == True:
            clean_text.append(edit)
        
        # extract token if it's a noun
        if token.tag_ in req_tag and token.shape_ != 'x' and token.shape_ != 'xx':
            extracted_words.append(token.lemma_)
            
    return clean_text, extracted_words


def apply_preprocessing(df):
    tqdm.pandas()
    df['reviewClean_sw'], df['noun_adjective'] = zip(*df['reviewText'].progress_apply(text_preprocessing))
    return df


# def apply_preprocessing(df):
#     processed_reviews = []
#     extracted_words = []
    
#     for review in tqdm(df['reviewText']):
#         processed_review, extracted_word = text_preprocessing(review)
#         processed_reviews.append(processed_review)
#         extracted_words.append(extracted_word)
    
#     df['reviewClean_sw'] = processed_reviews
#     df['noun_adjective'] = extracted_words

#     return df


# In[ ]:




