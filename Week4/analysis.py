#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:12:01 2020

@author: babaniyi
"""

'''
RESEARCH QUESTIONS

Our challenge for you: 
Let us know the relationship between a song being "explicit" and its other characteristics... 
One might assume being explicit would contribute to loudness, but how about popularity?

And if you wanna get really wild, how might you build a recommendation system using this data?
     - How would you learn a user's preferences over time? Is it a good idea to use features like
     energy or key? 
     
     - How can you tell what features are most important to individual users? Fascinating 
     questions that I'm sure Spotify's data team intimately understand
'''

'''
Variable Description: https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/

duration_ms - The duration of the track in milliseconds.
key - The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.

mode - Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.

time_signature - An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).

acousticness - A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. The distribution of values for this feature look like this: Acousticness distribution

danceability - Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. The distribution of values for this feature look like this: Danceability distribution

energy - Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. The distribution of values for this feature look like this: Energy distribution

instrumentalness - Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. The distribution of values for this feature look like this: Instrumentalness distribution

liveness - Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live. The distribution of values for this feature look like this: Liveness distribution

loudness - The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db. The distribution of values for this feature look like this: Loudness distribution

speechiness - Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. The distribution of values for this feature look like this: Speechiness distribution

valence - A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). The distribution of values for this feature look like this: Valence distribution

tempo - The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. The distribution of values for this feature look like this: Tempo distribution

id - The Spotify ID for the track.

type - The object type: “audio_features”

popularity - The popularity of the track. The value will be between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are. Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. Artist and album popularity is derived mathematically from track popularity. Note that the popularity value may lag actual popularity by a few days: the value is not updated in real time.


'''

'''
INDIVIDUAL DESCRIPTION

Primary:
- id (Id of track generated by Spotify)
Numerical:
- acousticness (Ranges from 0 to 1)
- danceability (Ranges from 0 to 1)
- energy (Ranges from 0 to 1)
- duration_ms (Integer typically ranging from 200k to 300k)
- instrumentalness (Ranges from 0 to 1)
- valence (Ranges from 0 to 1)
- popularity (Ranges from 0 to 100)
- tempo (Float typically ranging from 50 to 150)
- liveness (Ranges from 0 to 1)
- loudness (Float typically ranging from -60 to 0)
- speechiness (Ranges from 0 to 1)
- year (Ranges from 1921 to 2020)

Dummy:
- mode (0 = Minor, 1 = Major)
- explicit (0 = No explicit content, 1 = Explicit content)

Categorical:
- key (All keys on octave encoded as values ranging from 0 to 11, starting on C as 0, C# as 1 and so on…)
- artists (List of artists mentioned)
- release_date (Date of release mostly in yyyy-mm-dd format, however precision of date may vary)
- name (Name of the song)
'''


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('/Users/babaniyi/Documents/Tasks/Babaniyi/DailyDataset/Week4')

df = pd.read_csv('data.csv')
del df['Unnamed: 0']
df.dtypes



plt.figure(figsize=(16, 8))
sns.set(style="whitegrid")
corr = df.corr()
sns.heatmap(corr,annot=True,cmap="coolwarm")


#_____ Popularity of artists
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
x = df.groupby("artists")["popularity"].sum().sort_values(ascending=False).head(10)
ax = sns.barplot(x.index, x)
ax.set_title('Top Artists with Popularity')
ax.set_ylabel('Popularity')
ax.set_xlabel('Artists')
plt.xticks(rotation = 90)


#_____ Characteristics by time
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
columns = ["acousticness","danceability","energy","speechiness","liveness","valence"]
for col in columns:
    x = df.groupby("year")[col].mean()
    ax= sns.lineplot(x=x.index,y=x,label=col)
ax.set_title('Audio characteristics over year')
ax.set_ylabel('Measure')
ax.set_xlabel('Year')


#__________Explicitness
df[(df.explicit == 1) & (df.year == 1921)]

print(df['key'].nunique())


df['key'].value_counts()

explicit = pd.DataFrame(df.groupby('year')['explicit'].sum())
explicit = explicit.reset_index()
#explicit = explicit.sort_values('explicit', ascending = False)

ax = explicit.plot.bar(x='year', y='explicit', figsize=(20,10))


###################################

#________Recommending songs by title
from sklearn.feature_extraction.text import TfidfVectorizer # get the term frequency and inverse document frequency for calculating similarity scores

vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')

# Removing characters and numbers
df = pd.read_csv('data.csv')
del df['Unnamed: 0']


def clean_text(df, text_field, new_text_field):
    df[new_text_field] = df[text_field].str.lower()
    df[new_text_field] = df[new_text_field].apply(lambda elem: ''.join([i for i in elem if not i.isdigit()]))  
    #df[new_text_field] = df[new_text_field].apply(lambda x: ''.join([e for e in x if e.isalpha()]))
    return df

df = clean_text(df,'name', 'new_name')

df[['name','new_name']].head()

# Dropping songs with less popularity score <=10
songs_10 = df[df['popularity']>= 25]
#build book-title tfidf matrix
tfidf_matrix = vectorizer.fit_transform(songs_10['new_name'])
tfidf_feature_name = vectorizer.get_feature_names()
tfidf_matrix.shape


# computing cosine similarity matrix using linear_kernel of sklearn
from sklearn.metrics.pairwise import linear_kernel
cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

songs_10 = songs_10.reset_index(drop=True)
indices = pd.Series(songs_10['new_name'].index)
    
'''
################
#Function to get the most similar songs
def recommend(index, method):
    id = indices[index]
    # Get the pairwise similarity scores of all books compared that book,
    # sorting them and getting top 5
    similarity_scores = list(enumerate(method[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11]
    
    #Get the songs index
    songs_index = [i[0] for i in similarity_scores]
    
    #Return the top 5 most similar songs using integer-location based indexing (iloc)
    return songs_10['name'].iloc[songs_index]




# Let's recommend the following song
songs_10.iloc[150]

#input the index of the song
recommend(150, cosine_similarity)

##################
'''



def recommend_cosine(index):
    id = indices[index]
    # Get the pairwise similarity scores of all songs compared that song,
    # sorting them and getting top 10
    similarity_scores = list(enumerate(cosine_similarity[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11]
    
    #Get the books index
    songs_index = [i[0] for i in similarity_scores]
    
    #Return the top 5 most similar books using integer-location based indexing (iloc)
    return songs_10.iloc[songs_index]


recommend_cosine(150)

#____________ Calculating similarity score: Ecuclidean Distance

from sklearn.metrics.pairwise import euclidean_distances
D = euclidean_distances(tfidf_matrix)

def recommend_euclidean_distance(isbn):
    ind = indices[index]
    distance = list(enumerate(D[ind]))
    distance = sorted(distance, key=lambda x: x[1])
    distance = distance[1:6]
    #Get the books index
    books_index = [i[0] for i in distance]

    #Return the top 5 most similar books using integar-location based indexing (iloc)
    return books_wd.iloc[books_index]




df2 = df.drop_duplicates()








   
   