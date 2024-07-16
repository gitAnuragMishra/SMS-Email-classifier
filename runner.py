import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
word_corpus = stopwords.words('english')
from nltk.stem import PorterStemmer
port = PorterStemmer()
punc = string.punctuation
vect = pickle.load(open('vectoriser.pkl', 'rb'))
model = pickle.load(open('RFclassifier.pkl','rb'))

st.title('Spam Classifier')
text = st.text_area("Enter text")

def transform_text(text):
    text  = text.lower() #lowercasing
    text  = nltk.word_tokenize(text) #tokenize

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i) #remove non alphanumeric
    
    text = y.copy()
    y.clear()

    for i in text:
        if i not in word_corpus and i not in punc:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(port.stem(i))
    
    #return y #returns as a list
    return " ".join(y) #returns as a string

if st.button("predict"):

    transformed_text = transform_text(text)
    vector = vect.transform([transformed_text])
    result = model.predict(vector)


    if result == 1:
        st.header("Spam")

    else:
        st.header("Not Spam")

