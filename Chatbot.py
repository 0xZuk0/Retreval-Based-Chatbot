# Date : 20-09-2019
# Day : Friday
# Time : 8:50 pm
# The Code is written in Python language.



import numpy as np                                                 # importing modules
import pandas as pd                                                # numpy for arrays and other functions
from sklearn.feature_extrction.text import TfidfVectorizer          # sklearn for vectorizer and codine similarity btw words
from sklearn.metrics.pairwise import cosine_similarity             # pandas for reading csv file


# Reading csv file

df = pd.read_csv('aws_faq.csv')
df.dropna(inplace = True)

# Vectorization **Very Important step** #

vectorizer = TfidfVectorizer()
vectorizer.fit(np.concatenate((df.Question, df.Answer)))

 # Creating Vector Of Questions #
 
 Question_vectors = vectorizer.transform(df.Question)
 
 # Main Body #
 
 print('You Can Start Chatting with me Now.')
 
 while True :
      input_question = input()
      input_question_vector = vectorizer.transform([input_question])
      similarities = cosine_similarity(input_question_vector, Question_vectors)
      closest = np.argmax(similarities, axis = 1)
      
      print('BOT : ' + df.Answer.iloc[closest].values[0])
      

------------------------------------------------------END OF THE PROGRAM--------------------------------------------------------
