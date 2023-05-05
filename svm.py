import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

url_train = 'https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv'
df_train = pd.read_csv(url_train,index_col=0,parse_dates=[0],on_bad_lines='skip')
url_test = 'https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_test.csv'
df_test = pd.read_csv(url_test,index_col=0,parse_dates=[0],on_bad_lines='skip')

import nltk

df_train = pd.read_csv(url_train,index_col=0,parse_dates=[0],on_bad_lines='skip')
df_train = df_train.reset_index()
# 1. Removing Blank Spaces
df_train['medical_abstract'].dropna(inplace=True)
# 2. Changing all text to lowercase
df_train['medical_abstract_org'] = df_train['medical_abstract']
df_train['medical_abstract'] = [entry.lower() for entry in df_train['medical_abstract']]
# 3. Tokenization-In this each entry in the corpus will be broken into set of words
df_train['medical_abstract']= [word_tokenize(entry) for entry in df_train['medical_abstract']]
# 4. Remove Stop words, Non-Numeric and perfoming Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(df_train['medical_abstract']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    df_train.loc[index,'text_final'] = str(Final_words)

df_test = pd.read_csv(url_test,index_col=0,parse_dates=[0],on_bad_lines='skip')
df_test = df_test.reset_index()
# 1. Removing Blank Spaces
df_test['medical_abstract'].dropna(inplace=True)
# 2. Changing all text to lowercase
df_test['medical_abstract_org'] = df_test['medical_abstract']
df_test['medical_abstract'] = [entry.lower() for entry in df_test['medical_abstract']]
# 3. Tokenization-In this each entry in the corpus will be broken into set of words
df_test['medical_abstract']= [word_tokenize(entry) for entry in df_test['medical_abstract']]
# 4. Remove Stop words, Non-Numeric and perfoming Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(df_test['medical_abstract']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    df_test.loc[index,'text_final'] = str(Final_words)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(df_train['condition_label'])
Test_Y = Encoder.fit_transform(df_test['condition_label'])

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df_train['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(df_train['text_final'])
Test_X_Tfidf = Tfidf_vect.transform(df_test['text_final'])

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10], 'gamma': [1,0.1,0.01],'kernel': ['poly', 'sigmoid', 'rbf']}
# grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=2)
# grid.fit(Train_X_Tfidf,Train_Y)
# print(grid.best_estimator_)

SVM = svm.SVC(C=1.0, kernel='sigmoid', degree=3, gamma=1)
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

SVM_balance = svm.SVC(C=1.0, kernel='sigmoid', gamma=1, class_weight='balanced')
SVM_balance.fit(Train_X_Tfidf, Train_Y)
predictions_SVM_balance = SVM_balance.predict(Test_X_Tfidf)
print("SVM Accuracy Score Balanced -> ",accuracy_score(predictions_SVM_balance, Test_Y)*100)

# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=17)
# X_1, y_1 = smote.fit_resample(pd.DataFrame(df_train['medical_abstract']), df_train.condition_label)

# Tfidf_vect_1 = TfidfVectorizer(max_features=5000)
# Tfidf_vect_1.fit(df_train['text_final'])
# Train_X_Tfidf = Tfidf_vect.transform(df_train['text_final'])
# Test_X_Tfidf = Tfidf_vect.transform(df_test['text_final'])

# print(classification_report(Test_Y,predictions_SVM))
