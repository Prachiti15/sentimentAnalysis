import string
import re
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from bs4 import BeautifulSoup
import requests
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import pandas as pd
import os
import numpy as np
import pickle
import pandas as pd
# import wandb
import operator
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


from xgboost import XGBClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from yellowbrick.classifier import ClassificationReport
from yellowbrick.datasets import load_occupancy
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.model_selection import LearningCurve
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
module_dir = os.path.dirname(__file__)
file_path = os.path.join(module_dir, 'context.txt')  # full path to text.
data_file = open(file_path, 'r', encoding="utf8")
context = data_file.read()

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
cur_path = Path.cwd()
csv_path = str(cur_path)+"/Reviews.csv"
cd = pd.read_csv('cleaned_emotions.csv')
cd1 = pd.read_csv('cleaned_data.csv')

loaded_model1 = pickle.load(open('xgb.sav', 'rb'))

loaded_model = pickle.load(open('emotion_xgb_model.pkl', 'rb'))


def clean_text(text: str):
    text = str(text)
    text = text.lower()
    text = text.strip()
    text = re.sub(' \d+', ' ', text)
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)

    text = text.strip()

    return text


def remove_stopwords(text: str):
    text = str(text)
    filtered_sentence = []
    stop_words = ["a", "an", "the", "this", "that", "is", "it", "to", "and"]
    words = word_tokenize(text)
    for w in words:
        if w not in stop_words:
            filtered_sentence.append(w)
    text = " ".join(filtered_sentence)

    return text


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize(text: str):
    text = str(text)
    wl = WordNetLemmatizer()
    lemmatized_sentence = []
    words = word_tokenize(text)
    word_pos_tags = nltk.pos_tag(words)
    for idx, tag in enumerate(word_pos_tags):
        lemmatized_sentence.append(
            wl.lemmatize(tag[0], get_wordnet_pos(tag[1])))

    lemmatized_text = " ".join(lemmatized_sentence)

    return lemmatized_text
# Scraping starts here


def generate_url(product_name):
    base_url = "https://www.amazon.in/s?"
    search_query = product_name.replace(' ', '+')
    url = f"{base_url}k={search_query}&crid=3EDL9LLCO04NT&sprefix={search_query}%2Caps%2C336&ref=nb_sb_noss_2"
    return url


def getUrl(product_name):
    url = generate_url(product_name)

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0;Win64) AppleWebkit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
        # Add more User-Agent strings here
    ]

    max_retries = 5
    retry_delay = 1  # seconds

    asins = []

    for user_agent in user_agents:
        headers = {
            "User-Agent": user_agent
        }

        for retry in range(max_retries):
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")
                results = soup.find_all("div", {"data-csa-c-item-id": True})
                for result in results:
                    asin = result["data-csa-c-item-id"].split(".")
                    asins.append(asin[-1])

                # Break the loop if successful
                break

            except requests.exceptions.RequestException as e:
                # Retry after a delay
                time.sleep(retry_delay)

        # Break the loop if ASINs are retrieved successfully
        if asins:
            reviews = getReview2(asins)
            return reviews


# def getReview(asins):
#     final_score = []
#     k = 1
#     loop_len = 10
#     if len(asins) < 10:
#         loop_len = len(asins)
#     for i in range(loop_len):
#         url = "https://www.amazon.in/product-reviews/{}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews".format(
#             asins[i])
#         page = requests.get(url)
#         soup = BeautifulSoup(page.content, 'html.parser')
#         review = soup.select("span.review-text-content span")
#         cust_review = []
#         for i in range(0, len(review)):
#             cust_review.append(review[i].get_text())
#         df = pd.DataFrame()
#         df["review"] = cust_review
#         pathTocsv = "./review" + str(i)
#         df.to_csv(pathTocsv)
#         df['Review'] = df['review'].apply(clean_text)
#         df['Review'] = df['Review'].apply(remove_stopwords)
#         df['Review'] = df['Review'].apply(lemmatize)
#         df.to_csv(csv_path, index=False)
#         data1 = pd.read_csv("Reviews.csv")
#         if len(data1['Review']) == 0:
#             print("Request Denied from Amazon! Moving on to next product")
#         else:
#             productName = soup.find(
#                 'a', {'data-hook': 'product-link'}).text.strip()
#             stars1 = soup.find('i', {'class': 'averageStarRating'})
#             starRating1 = float(stars1.text.strip().split(' ')[0])
#             data1[data1['Review'].isnull()]
#             data1.dropna(inplace=True)
#             vectorizer1 = TfidfVectorizer(max_features=1000)
#             vectorizer1.fit(cd['text'])
#             features1 = vectorizer1.transform(data1['text'])
#             features1.toarray()
#             tf_idf1 = pd.DataFrame(features1.toarray(),
#                                    columns=vectorizer1.get_feature_names_out())
#             ypred1 = loaded_model1.predict(tf_idf1)
#             data1['Review'].to_csv(str(cur_path)+"/temp.csv")
#             fscr1 = 0
#             for j in ypred1:
#                 if j == 1:
#                     fscr1 = fscr1 + 1
#             scr1 = (fscr1)/len(ypred1)
#             fs1 = round((map_values(scr1) + starRating1)/2, 2)
#             final_score.append(fs1)
#             # print('{} product processed'.format(k))
#             k = k + 1
#             ypred1 = []
#         time.sleep(3)
#     return final_score


def map_values(value):
    mapped_value = value * (5.0 - 1.0) + 1.0
    return mapped_value


# New Model Starts Here
# satis - 0
# happy - 1
# mixed - 2
# dissapointed - 3

def getReview2(asins):
    final_score = []
    k = 1
    temp = 0
    loop_len = 10
    if len(asins) < 10:
        loop_len = len(asins)
    for i in range(loop_len):
        url = "https://www.amazon.in/product-reviews/{}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews".format(
            asins[i])
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        review = soup.select("span.review-text-content span")
        cust_review = []
        for i in range(0, len(review)):
            cust_review.append(review[i].get_text())
        df = pd.DataFrame()
        df["review"] = cust_review
        pathTocsv = "./review" + str(i)
        df.to_csv(pathTocsv)
        df['Review'] = df['review'].apply(clean_text)
        df['Review'] = df['Review'].apply(remove_stopwords)
        df['Review'] = df['Review'].apply(lemmatize)
        df.to_csv(csv_path, index=False)
        data = pd.read_csv("Reviews.csv")
        data1 = pd.read_csv("Reviews.csv")
        if len(data['Review']) == 0:
            print("Request Denied from Amazon! Moving on to next product")
        else:
            # NEW MODEL
            productName = soup.find(
                'a', {'data-hook': 'product-link'}).text.strip()
            stars = soup.find('i', {'class': 'averageStarRating'})
            starRating = float(stars.text.strip().split(' ')[0])
            data[data['Review'].isnull()]
            data.dropna(inplace=True)
            vectorizer = TfidfVectorizer(max_features=1000)
            vectorizer.fit(cd['text'])
            features = vectorizer.transform(data['Review'].values.astype('U'))
            features.toarray()
            tf_idf = pd.DataFrame(features.toarray(),
                                  columns=vectorizer.get_feature_names_out())
            ypred = loaded_model.predict(tf_idf)
            fscr = 0
            for sc in ypred:
                if sc == 0:
                    fscr = fscr + 1
                if sc == 1:
                    fscr = fscr + 1
                if sc == 3:
                    fscr = fscr - 1
            scr = (fscr)/len(ypred)
            fs = round((map_values(scr) + starRating)/2, 2)
            data['Review'].to_csv(str(cur_path)+"/temp.csv")
            final_score.append(
                [productName, fs, starRating, ypred.tolist()])
            temp = temp + 1
            print('{} product processed'.format(k))
            k = k + 1
            ypred = []
            # print(ypred)
            # fscr = 0
            # for j in ypred:
            #     if j == 1:
            #         fscr = fscr + 1
            # scr = (fscr)/len(ypred)
            # fs = round((map_values(scr)+starRating)/2, 2)
            # Old model
            # if len(data['Review']) == 0:
            #     print("Request Denied from Amazon! Moving on to next product")
            # else:
            #     vectorizer1 = TfidfVectorizer(max_features=1000)
            #     vectorizer1.fit(cd1['text'])
            #     features1 = vectorizer1.transform(
            #         data1['Review'].values.astype('U'))
            #     features1.toarray()
            #     tf_idf1 = pd.DataFrame(features1.toarray(),
            #                            columns=vectorizer1.get_feature_names_out())
            #     ypred1 = loaded_model1.predict(tf_idf1)
            #     # print(f"The predictions from the old model are {ypred1}")

            #     # for j in ypred1:
            #     #     if j == 1:
            #     #         fscr = fscr + 1
            #     # scr = (fscr)/len(ypred)
            #     fs = round((map_values(scr) + starRating)/2, 2)
            #     # print('{} product processed'.format(k))
            #     ypred1 = []
            #     final_score.append(
            #         [productName, fs, starRating, ypred.tolist()])
            #     temp = temp + 1
            #     print('{} product processed'.format(k))
            #     k = k + 1
            #     ypred = []
        time.sleep(1)
    final_score.sort(key=operator.itemgetter(1), reverse=True)
    order = 0

    for a in range(len(final_score)):
        final_score[a].append(order)
        order = order + 1
    # print(final_score)
    return final_score
