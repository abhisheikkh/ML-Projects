#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install numpy')


# In[6]:


get_ipython().system('pip install pandas')


# In[7]:


import numpy as np
import pandas as pd


# In[8]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[9]:


movies.head()


# In[10]:


credits.head()


# In[11]:


movies = movies.merge(credits, on='title')


# In[12]:


movies.head(1)


# In[13]:


#drop unwanted columns

movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[14]:


movies.head()


# In[15]:


movies.isnull().sum()


# In[16]:


movies.dropna(inplace=True)


# In[17]:


movies.isnull().sum()


# In[18]:


movies.duplicated().sum()


# In[19]:


movies.iloc[0].genres


# In[20]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
#['Action','Adventure','Fantasy','SciFi']


# In[21]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[22]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[23]:


movies['genres']=movies['genres'].apply(convert)


# In[24]:


movies


# In[25]:


movies['keywords']=movies['keywords'].apply(convert)


# In[26]:


movies.head()


# In[27]:


movies['cast'][0]


# In[28]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1;
        else:
            break
    return L


# In[29]:


movies['cast']=movies['cast'].apply(convert3)


# In[30]:


movies.head()


# In[31]:


movies['crew'][0]


# In[32]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=="Director":
            L.append(i['name'])
            break
    return L


# In[33]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[34]:


movies.head()


# In[35]:


movies['overview'][0]


# In[36]:


#convert overview to a list
movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[37]:


movies.head()


# In[38]:


#remove spaces between words so that confusion can be removed. eg. samworthington & sammendes 


# In[39]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[40]:


movies


# In[41]:


movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[42]:


movies


# In[43]:


#create a column tag by concatenating overview, genres, keywords, cast and crew


# In[44]:


movies['tags']= movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[45]:


movies


# In[46]:


new_df = movies[['movie_id','title','tags']]


# In[47]:


new_df


# In[48]:


#coverting tags from list to string
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[49]:


new_df.head()


# In[50]:


new_df['tags'][0]


# In[51]:


new_df['tags']=new_df['tags'].apply(lambda x: x.lower())


# In[52]:


new_df.head()


# In[68]:


get_ipython().system('pip install scikit-learn')


# In[69]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')


# In[70]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[71]:


cv.fit_transform(new_df['tags']).toarray().shape


# In[72]:


vectors


# In[73]:


cv.get_feature_names_out()


# In[74]:


#we need to treat all words' similar forms as one eg. loving loved as love. use stemming for that


# In[75]:


get_ipython().system('pip install nltk')


# In[60]:


import nltk


# In[76]:


from nltk.stem.porter import PorterStemmer
ps =  PorterStemmer()


# In[77]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[78]:


ps.stem('loving')


# In[79]:


ps.stem('loved')


# In[80]:


new_df['tags']=new_df['tags'].apply(stem)


# In[81]:


#now as we have vectorized tags, we then calculate the cosine distance btw
#all the movies from every other one
#shorter the distance, more similar the movies are to each other


# In[82]:


vectors


# In[83]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[84]:


from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(vectors)


# In[85]:


similarity=cosine_similarity(vectors)


# In[86]:


cosine_similarity(vectors).shape


# In[87]:


similarity[0]


# In[88]:


#enumerate to bring the index num along with it, otherwise we lose it while sorting
#sorting from first position and not second
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[89]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances =  similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(i)


# In[90]:


recommend('Batman Begins')


# In[91]:


import pickle


# In[92]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[93]:


new_df['title'].values


# In[99]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[100]:


pickle.dump(similarity, open('similarity.pkl','wb'))


# In[ ]:




