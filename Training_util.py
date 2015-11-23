# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:26:16 2015

@author: Adithya
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 08:33:33 2015

@author: Adithya
"""
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.externals import joblib

def extractDesiredFeatures(data,train):
    extractedData=pd.DataFrame();
    extractedData["BodyLength"]=data["BodyMarkdown"].apply(len)
    extractedData["TitleLength"]=data["Title"].apply(len)
    extractedData["ReputationAtPostCreation"]=data["ReputationAtPostCreation"]
    extractedData["OwnerUndeletedAnswerCountAtPostTime"]=data["OwnerUndeletedAnswerCountAtPostTime"]
    extractedData["NumOfTags"]=[sum(map(lambda x:pd.isnull(x),row)) for row in data[['Tag1','Tag2','Tag3','Tag4','Tag5']].values]
    extractedData["NumOfTags"]=5-extractedData["NumOfTags"];
    extractedData["UserAge"]=pd.to_datetime(data["PostCreationDate"])
    extractedData["UserAge"]=extractedData["UserAge"]-pd.to_datetime(data["OwnerCreationDate"])
    extractedData["UserAge"]=extractedData["UserAge"].astype('timedelta64[s]')
    #BoWData=[]
    #BoWData=BagOfWord(data)
    #extractedData=pd.concat([extractedData,BoWData],axis=1)
    
#    if (train == 1):    
#        extractedData["Target"]=

    #extractedData=np.array(extractedData)
    #extractedData=np.concatenate((extractedData,BoWData),axis=1)    
    return (extractedData)

def GetTarget(data):
    target=[]
    target=data["OpenStatus"].eq('open')
    return (target)


def BagOfWordT(data):    
    cleaned_data=np.zeros((len(data),1),dtype=str)
    cleaned_data=pd.DataFrame(cleaned_data,columns=['cleanedBody']);
    cleaned_data
    for i in range(0,len(data)):
        cleaned_data["cleanedBody"][i]=review_to_words(data["BodyMarkdown"][i])
    
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = 'english',   \
                                 max_features = 1000) 
   
    
    vectorizer1= vectorizer.fit(np.array(data["Tag1"],dtype=str))
    joblib.dump(vectorizer1, "C:/Users/Adithya/Desktop/OpenClosed/vec1/vec1.pkl")
    train_data_features=vectorizer1.transform(cleaned_data["cleanedBody"])    
    return (train_data_features)
def BagOfWordQT(data):    
    cleaned_data=np.zeros((len(data),1),dtype=str)
    #stop_data=np.zeros((len(data),1),dtype=int)
    #stop_data=pd.DataFrame(stop_data,columns=['Stop_data']);
    cleaned_data=pd.DataFrame(cleaned_data,columns=['cleanedBody']);
    cleaned_data
    for i in range(0,len(data)):
        #init_len=len(data["Title"][i])
        cleaned_data["cleanedBody"][i]=review_to_words(data["Title"][i])
        #stop_data["stop_data"][i]=init_len-len(cleaned_data["cleanedBody"][i])
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = 'english',   \
                                 max_features = 1000) 
    
    
    
    vectorizer1= vectorizer.fit(np.array(data["Tag1"],dtype=str))
    train_data_features=vectorizer1.transform(cleaned_data["cleanedBody"])    
    return (train_data_features)
def BagOfWord(data):    
    cleaned_data=np.zeros((len(data),1),dtype=str)
    cleaned_data=pd.DataFrame(cleaned_data,columns=['cleanedBody']);
    cleaned_data
    for i in range(0,len(data)):
        cleaned_data["cleanedBody"][i]=review_to_words(data["BodyMarkdown"][i])
    
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = 'english',   \
                                 max_features = 1000) 
    
   
    vectorizer2=vectorizer.fit(np.array(cleaned_data["cleanedBody"],dtype=str))
    joblib.dump(vectorizer2, "C:/Users/Adithya/Desktop/OpenClosed/vec2/vec2.pkl")

    train_data_features = vectorizer2.transform(list(cleaned_data["cleanedBody"]))
    return (train_data_features)
def BagOfWordQ(data):    
    cleaned_data=np.zeros((len(data),1),dtype=str)
    #stop_data=np.zeros((len(data),1),dtype=int)
    #stop_data=pd.DataFrame(stop_data,columns=['Stop_data']);
    cleaned_data=pd.DataFrame(cleaned_data,columns=['cleanedBody']);
    cleaned_data
    for i in range(0,len(data)):
        #init_len=len(data["Title"][i])
        cleaned_data["cleanedBody"][i]=review_to_words(data["Title"][i])
        #stop_data["stop_data"][i]=init_len-len(cleaned_data["cleanedBody"][i])
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = 'english',   \
                                 max_features = 1000) 
    
    
    vectorizer3=vectorizer.fit(np.array(cleaned_data["cleanedBody"],dtype=str))
    joblib.dump(vectorizer3, "C:/Users/Adithya/Desktop/OpenClosed/vec3/vec3.pkl")

    train_data_features = vectorizer3.transform(list(cleaned_data["cleanedBody"]))
    return (train_data_features)

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    #review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops =      ["a", "able", "about", "across", "after", "all", "almost", "also", "am",

                "among", "an","and","any","are","as","at","be","because","been","but","by","can",

                "cannot","could","dear","did","do","does","either","else","ever","every",
                
                "for","from","get","got","had","has","have","he","her","hers","him","his",
                
                "how","however","i","if","in","into","is","it","its","just","least","let",
                
                "like","likely","may","me","might","most","must","my","neither","no","nor",
                
                "not","of","off","often","on","only","or","other","our","own","rather","said",
                
                "say","says","she","should","since","so","some","than","that","the","their",
                
                "them","then","there","these","they","this","tis","to","too","twas","us",
                
                "wants","was","we","were","what","when","where","which","while","who",
                
                "whom","why","will","with","would","yet","you","your","?"]                 
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))
def FindTagFreq(data):
    
    compressdata=np.zeros((len(data),1),dtype=str)
    compressdata=pd.DataFrame(compressdata,columns=['cleanedBody']);    
    compressdata=data["Tag1"]+" "+data["Tag2"]+" "+data["Tag3"]+" "+data["Tag4"]+" "+data["Tag5"]
    compressdata=compressdata.replace(np.nan,'None')
    for i in range(0,len(data)):
        #init_len=len(data["Title"][i])
        compressdata[i]=review_to_words(compressdata[i])
        
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = 'english',   \
                                 max_features = 1000) 
    
    
    
    vectorizer4= vectorizer.fit(np.array(compressdata,dtype=str))
    joblib.dump(vectorizer4, "C:/Users/Adithya/Desktop/OpenClosed/vec4/vec4.pkl")
    
    train_data_features=vectorizer4.transform(compressdata)    
    return (train_data_features)
    
data=pd.read_csv("C:/Users/Adithya/Desktop/OpenClosed/train-sample.csv")
initial_features=extractDesiredFeatures(data,1)
word_vectorsT=BagOfWordT(data)
word_vectorsQT=BagOfWordQT(data)
word_vectors=BagOfWord(data)
word_vectorsQ=BagOfWordQ(data)

tag_freq=FindTagFreq(data)

initial_features=sp.sparse.csr_matrix(initial_features)
initial_features=sp.sparse.hstack((initial_features,word_vectorsT,word_vectorsQT,word_vectors,word_vectorsQ,tag_freq))
initial_features=sp.sparse.csr_matrix(initial_features)
target=GetTarget(data)


