#importing dependencies
import numpy as np 
import pandas as pd
import ast 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
 

movies = pd.read_csv("./movies.csv")
credits = pd.read_csv("./credits.csv")


movies = movies.merge(credits,on='title')
#print(movies.head(1))

#important columns for creating tags 
#genres
#id
#keywords
#title
#overview
#cast
#crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


movies.dropna(inplace=True)
#print(movies.iloc[0].genres)

#function to fetch genre of movies

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

#fucntion to pick top 3 actors of the movie 

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
        
    return L 

movies['cast'] = movies['cast'].apply(convert3)


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


movies['crew'] = movies['crew'].apply(fetch_director)

        



movies['overview'] = movies['overview'].apply(lambda x:x.split())

#removing whitespaces from diferent tags
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

#let's create a new column with name tag in which we will concatenate all different columns required for tags 
movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

#now let's create a new datafram with movies , id and tags column 

new_df = movies[['movie_id','title','tags']]

new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
#print(new_df['tags'])


#let's make an object of countvectorizer 

cv = CountVectorizer(max_features=5000,stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

#create an object of porter-stemmer

ps = PorterStemmer()

#create a helper function for stemming

def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

similarity  = cosine_similarity(vectors)

#function to find most similar movie 

def recommend(movie):
    movie_index = new_df[new_df['title']== movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        
#recommend('Batman Begins')


#create a pickle file 

pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))







        


