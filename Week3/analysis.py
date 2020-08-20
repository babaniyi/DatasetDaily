#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:41:43 2020

@author: babaniyi
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
import nlp
import joblib

os.chdir('/Users/babaniyi/Documents/Tasks/Babaniyi/DailyDataset/Week3')



'''
list of datasets in the package
print([dataset.id for dataset in nlp.list_datasets()])
'''
news_dataset = nlp.load_dataset('civil_comments')

#test_data = pd.DataFrame.from_dict(news_dataset['test'])
train_data = pd.DataFrame.from_dict(news_dataset['train'])
#val_data = pd.DataFrame.from_dict(news_dataset['validation'])


train_data.dtypes
train_data.describe()


'''
def compute_log_loss(predicted, actual, eps = 1e-14):
    predicted = np.clip(predicted,eps, 1-eps)
    loss = - 1 * np.mean(actual + np.log(predicted)) + (1 - actual) * np.log(1-predicted)
    return loss
'''

train_data['new_toxicity'] = np.where(train_data['toxicity']>=0.5,1,0)
train_data['new_severe_toxicity'] = np.where(train_data['severe_toxicity']>=0.6,1,0)
train_data['new_obscene'] = np.where(train_data['obscene']>=0.5,1,0)
train_data['new_threat'] = np.where(train_data['threat']>=0.5,1,0)
train_data['new_insult'] = np.where(train_data['insult']>=0.5,1,0)
train_data['new_identity_attack'] = np.where(train_data['identity_attack']>=0.5,1,0)
train_data['new_sexual_explicit'] = np.where(train_data['sexual_explicit']>=0.5,1,0)



def icount(data):
    print(data.value_counts())
    
icount(train_data['new_toxicity'])
icount(train_data['new_severe_toxicity'])
icount(train_data['new_obscene'])
icount(train_data['new_threat'])
icount(train_data['new_insult'])
icount(train_data['new_identity_attack'])
icount(train_data['new_sexual_explicit'])


# _____________________Removing characters

import re

def clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    return df

train_data = clean_text(train_data, "text", "text_clean")
train_data['text_clean'].head(5)

#_______________Removing Stopwords such as the, it, as
import nltk.corpus
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
stop.extend(['the', 'that', 'but', 'they', 'and', 'you', 'we', 'he', 'this', 'not', 'what', 'so',
             'I\'m', 'are', 'if', 'to', 'said', 'say', 'don\'t'])

train_data['text_clean'] = train_data['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
train_data['text_clean'].head(10)

#_______________Lemmatizing
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def word_lemmatizer(text):
    lem_text = [WordNetLemmatizer().lemmatize(i) for i in text]
    return lem_text

'''
train_data['text_tokens'] = train_data['text_clean'].apply(lambda x: word_tokenize(x))

train_data['text_tokens_lemma'] = train_data['text_tokens'].apply(lambda x: word_lemmatizer(x))
train_data['text_tokens_lemma'].head(10)

train_data['text'] = train_data['text_tokens_lemma'].apply(lambda x: ' '.join(x))
train_data['text'].head()
'''

train_data['text'] = train_data['text'].apply(lambda x: ' '.join([word for word in word_lemmatizer(x.split())]))


#________________Number of words
train_data['word_count'] = train_data['text'].apply(lambda x: len(str(x.split())))    
train_data['word_count'].head()



#_______________ Part of Speech Counts
'''
     
Frequency distribution of Part of Speech Tags:
    Noun Count
    Verb Count
    Adjective Count
    Adverb Count
    Pronoun Count
     
'''

from textblob import TextBlob

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

train_data['noun_count'] = train_data['text'].apply(lambda x: check_pos_tag(x, 'noun'))
train_data['verb_count'] = train_data['text'].apply(lambda x: check_pos_tag(x, 'verb'))
train_data['adj_count'] = train_data['text'].apply(lambda x: check_pos_tag(x, 'adj'))
train_data['adv_count'] = train_data['text'].apply(lambda x: check_pos_tag(x, 'adv'))
train_data['pron_count'] = train_data['text'].apply(lambda x: check_pos_tag(x, 'pron'))


###########################
#_________ Generate a Word Cloud

sample_data = train_data[:10000]


# Import the wordcloud library
from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(list(sample_data['text'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()



# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer


sns.set_style('whitegrid')

# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('Words')
    plt.ylabel('Frequency of word')
    #plt.savefig('common_10.png')
    plt.show()
    
    
    
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(sample_data['text'])
# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)


######################
'''
Techniques:
1. Use only alphanumeric as tokens
2. N-gram range: i. apply dimension reduction and select best only best features
                ii. scaling: scaling relevant features using MaxAbsScaler
3. Include interaction terms: use SparseInteractions() for computation ease
4. Use Hashing: for memory efficiency
5. grid search: hyp tuning selcting the best C in logistic regresssion
'''

from SparseInteractions import SparseInteractions
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

train_data = train_data.drop(['text_clean', 'text_tokens', 'text_tokens_lemma'], axis=1)

get_text_data = FunctionTransformer(lambda x: x['text'], validate =False)
get_text_data.fit_transform(train_data.head(5))

get_numeric_data = FunctionTransformer(lambda x: x['word_count'], validate =False)
get_numeric_data.fit_transform(train_data.head(5))

alpha_tk = '[A-zA-z0-9]+(?=\\s+)'
chi_k = 300


numeric_features = ['word_count','noun_count','verb_count','adj_count','adv_count','pron_count']

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])

text_features = ['text']

text_transformer = Pipeline([
        ('selector', get_text_data),
        ('Vectorizer', HashingVectorizer(token_pattern=alpha_tk, ngram_range=(1,2), alternate_sign=False)),
        ('dim_red',SelectKBest(chi2,chi_k))
        ])

preprocessor = ColumnTransformer(
    transformers=[
            ('num', numeric_transformer, numeric_features),
            ('txt', text_transformer, text_features)])


clfr5 = Pipeline(steps=[('preprocessor', preprocessor),
                       ('int', SparseInteractions(degree=2)),
                       ('sc', MaxAbsScaler()),
                      ('classifier', OneVsRestClassifier(LogisticRegression(C=10, solver = 'lbfgs')))])

target_na = ['new_toxicity', 'new_severe_toxicity', 'new_obscene', 'new_threat',
       'new_insult', 'new_identity_attack', 'new_sexual_explicit']

feat_na = ['text','word_count','noun_count','verb_count','adj_count','adv_count','pron_count']


clfr5.fit(train_data[feat_na], train_data[target_na])
tr_pred5 = clfr5.predict(train_data[feat_na])
tr_pred5_prob = clfr5.predict_proba(train_data[feat_na])


from sklearn.metrics import classification_report
#print("model score: %.3f" % clfr.score(X,Y)) 

print(classification_report(train_data[target_na], tr_pred5, target_names = target_na))


test_data 
test_data = clean_text(test_data, "text", "text_clean")
test_data['text'] = test_data['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

test_data['word_count'] = test_data['text'].apply(lambda x: len(str(x.split())))    

test_data['noun_count'] = test_data['text'].apply(lambda x: check_pos_tag(x, 'noun'))
test_data['verb_count'] = test_data['text'].apply(lambda x: check_pos_tag(x, 'verb'))
test_data['adj_count'] = test_data['text'].apply(lambda x: check_pos_tag(x, 'adj'))
test_data['adv_count'] = test_data['text'].apply(lambda x: check_pos_tag(x, 'adv'))
test_data['pron_count'] = test_data['text'].apply(lambda x: check_pos_tag(x, 'pron'))


test_data['new_toxicity'] = np.where(test_data['toxicity']>=0.5,1,0)
test_data['new_severe_toxicity'] = np.where(test_data['severe_toxicity']>=0.6,1,0)
test_data['new_obscene'] = np.where(test_data['obscene']>=0.5,1,0)
test_data['new_threat'] = np.where(test_data['threat']>=0.5,1,0)
test_data['new_identity_attack'] = np.where(test_data['identity_attack']>=0.5,1,0)
test_data['new_sexual_explicit'] = np.where(test_data['sexual_explicit']>=0.5,1,0)
test_data['new_insult'] = np.where(test_data['insult']>=0.5,1,0)


pred5_prob = clfr5.predict(test_data[feat_na])
print(classification_report(test_data[target_na], pred5_prob, target_names = target_na))

from sklearn.metrics import mean_squared_error
from math import sqrt

p_prob = clfr5.predict_proba(test_data[feat_na])

pred_df = pd.DataFrame(columns = target_na, index = test_data.index, data = p_prob)

print(sqrt(mean_squared_error(test_data['toxicity'], pred_df['new_toxicity'])))
print(sqrt(mean_squared_error(test_data['severe_toxicity'], pred_df['new_severe_toxicity'])))
print(sqrt(mean_squared_error(test_data['obscene'], pred_df['new_obscene'])))
print(sqrt(mean_squared_error(test_data['threat'], pred_df['new_threat'])))
print(sqrt(mean_squared_error(test_data['insult'], pred_df['new_insult'])))
print(sqrt(mean_squared_error(test_data['identity_attack'], pred_df['new_identity_attack'])))
print(sqrt(mean_squared_error(test_data['sexual_explicit'], pred_df['new_sexual_explicit'])))






'''
#___________________________________________
# HYPERPARAMETER TUNING
from sklearn.model_selection import GridSearchCV


clfr1 = Pipeline(steps=[('preprocessor', preprocessor),
                       ('int', SparseInteractions(degree=2)),
                       ('sc', MaxAbsScaler()),
                      ('classifier', OneVsRestClassifier(LogisticRegression()))])


p_grid = {
    'classifier__estimator__C': [0.1, 1.0, 10, 100]
}


grid_search = GridSearchCV(clfr1, param_grid = p_grid, cv=3)
grid_search.fit(train_data[feat_na], train_data[target_na])

clfr1.fit(X, Y)
tr_pred1 = clfr1.predict(X)
print(classification_report(Y, tr_pred1, target_names = target_na))




#__________ XGB__________
import xgboost as xgb

clfr2 = Pipeline(steps=[('preprocessor', preprocessor),
                       ('int', SparseInteractions(degree=2)),
                       ('sc', MaxAbsScaler()),
                      ('classifier', OneVsRestClassifier(xgb.XGBClassifier(seed = 42)))])

clfr2.fit(X, Y)

tr_pred2 = clfr2.predict(X)

print(classification_report(Y, tr_pred1, target_names = target_na))
'''

#___________________________________________
#___________________________________________
#___________________________________________
#___________________________________________

val_data['text'] = clean_text(val_data, "text")
val_data['text'] = val_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
val_data['text'] = val_data['text'].apply(lambda x: ' '.join([word for word in word_lemmatizer(x.split())]))
val_data['word_count'] = val_data['text'].apply(lambda x: len(str(x.split())))    


tr_pred2 = clfr1.predict(val_data[['text','word_count']])

val_data['new_toxicity'] = np.where(val_data['toxicity']>=0.5,1,0)
val_data['new_severe_toxicity'] = np.where(val_data['severe_toxicity']>=0.6,1,0)
val_data['new_obscene'] = np.where(val_data['obscene']>=0.5,1,0)
val_data['new_threat'] = np.where(val_data['insult']>=0.5,1,0)
val_data['new_identity_attack'] = np.where(val_data['identity_attack']>=0.5,1,0)
val_data['new_sexual_explicit'] = np.where(val_data['sexual_explicit']>=0.5,1,0)
val_data['new_insult'] = np.where(val_data['insult']>=0.5,1,0)


cols = ['new_toxicity',
       'new_severe_toxicity', 'new_obscene', 'new_threat',
       'new_identity_attack', 'new_sexual_explicit', 'new_insult']

print(classification_report(val_data[cols], tr_pred1, target_names = target_na))



#########################
###############################################################################
##############




#########################
###############################################################################
##############


