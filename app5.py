#Importing Libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit as st


#Load the Dataset
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
st.title("Sentiment Analysis on Restaurant Reviews")
st.markdown("---",unsafe_allow_html=True)
st.write(df.head())




#Cleaning the reviews

corpus = []
ps = PorterStemmer()

for i in range(0,df.shape[0]):
    
    #Cleaning special character from the reviews
    #The re.sub() method performs global search and global replace on the given string
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df.Review[i])
    
    #Converting the entire message into lower case
    message = message.lower()
    
    # Tokenizing the review by words
    words = message.split()  
    
    #Removing the stop words
    words = [word for word in words if word not in set(stopwords.words('english'))]
    
    #Stemming the words
    words = [ps.stem(word) for word in words] 
    
    #Joining the stemmed words
    message = ' '.join(words)
    
    #Building a corpus of messages
    corpus.append(message) 
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = df.iloc[:, 1].values
    
    
#Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


#Model
classifier = MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)


#Prediction
def predict_review(user_data):
    sample_message = re.sub(pattern='[^a-zA-Z]',repl=' ', string = user_data)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_message = [ps.stem(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    temp = cv.transform([final_message]).toarray()
    return classifier.predict(temp)


user_data=st.text_input(label='Feedback of our restaurant:')
st.subheader('Review stats')
st.write(user_data)

st.subheader("Accuracy")
acc_s = accuracy_score(y_test, classifier.predict(X_test))*100
st.write("Accuracy Score {} %".format(round(acc_s,2)))



st.subheader('Final Review: ')
result = ['Ohhh Shit! Negetive Review👎🏻','Good! Positive Review🤩👏🏻']

if predict_review(user_data):
    st.write(result[1])
else:
    st.write(result[0])
