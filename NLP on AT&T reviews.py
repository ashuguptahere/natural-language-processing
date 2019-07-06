# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 01:03:51 2019

@author: Aashish Gupta
"""

# Imporing Essestial Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
import csv

# Coping all Links into variable 'links'
with open('Links.csv') as f:
    reader = csv.reader(f)
    links = [row[0] for row in reader]

from selenium import webdriver
driver = webdriver.Chrome(executable_path='/chromedriver.exe')

reviews = []
for link in links:
    response = driver.get(link)
    html_source = driver.page_source
    soup = BeautifulSoup(html_source, 'html.parser')
    i = soup.find_all('div', {'class':'reviewSnippetCard'})
    for j in i:
        review = j.find('p', {'class':'snippetSummary'}).text
        rating = int(j.find('span', {'class': 'stars-icon-star-on'})['style'][6:-2])/20
        reviews.append([review,rating])

dataset = pd.DataFrame(reviews, columns=['Review', 'Rating'])

dataset.to_csv('reviews.csv', index=False)

# Importing some other Libraries
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# Preprocessing the data for NLP
processed_reviews = []
for i in range(len(dataset)):    
    review = re.sub('@[\w]*', ' ', dataset['Review'][i])
    review = re.sub('^a-zA-Z#', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(token) for token in review if not token in stopwords.words('english')]
    review = ' '.join(review)
    processed_reviews.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)

# Defining X (Feature Matrix) and y (Vector of Predictions)
X = cv.fit_transform(processed_reviews).toarray()
y = dataset['Rating'].values

# Making y in binary(either 0 or 1)
for i in range(len(y)):
    if y[i] >= 4:
        y[i] = 1 # 1 means Positive
    else:
        y[i] = 0 # 0 means Negative

# Cutting X and y into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Printing all 
print(cv.get_feature_names())

# Creating a Guassian Naive Bayes Object
from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()

# Checking Score on Train Set
n_b.fit(X_train, y_train)
n_b.score(X_train, y_train)

# Checking Score on Test Set
n_b.fit(X_test, y_test)
n_b.score(X_test, y_test)

# Checking Score on Actual Data
n_b.fit(X, y)
n_b.score(X, y)

# If we pass a new review then:
z = 'I love their network'

review = re.sub('@[\w]*', ' ', z)
review = re.sub('^a-zA-Z#', ' ', review)
review = review.lower()
review = review.split()
review = [ps.stem(token) for token in review if not token in stopwords.words('english')]
review = ' '.join(review)
processed_reviews.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = len(X.transpose()))
X1 = cv.fit_transform(processed_reviews).toarray()

n_b.predict(X1[[-1]])