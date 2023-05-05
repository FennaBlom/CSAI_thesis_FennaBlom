import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer

Encoder = LabelEncoder()


def CreateData(url):
    """
    Returns the text values and labels of an online training dataset

        Parameters:
            url (str): URL to the training data (csv)
        
        Returns:
            texts (array): Array of all the texts in the dataset
            Y (ndarray): Array with the encoded labels of all abstracts
    """    
    train = pd.read_csv(url,index_col=0,parse_dates=[0],on_bad_lines='skip')
    train = train.reset_index()
    texts = train.medical_abstract.values
    Y = Encoder.fit_transform(train.condition_label)
    return texts, Y, Encoder

def Tokenization(texts, model_name):
    """
    Returns the tokenized text data

        Parameters:
            texts (ndarray): Array of all texts in dataset
            model_name (str): Name of pre-trained model
        
        Returns:
            input_ids (lst): List with encoded input ids
            attention_masks (lst): List with attention masks #rephrase?
    """ 
    print('Preprocessing data ...')
    texts_2 = []
    for entry in texts:
        abstract = word_tokenize(entry)
        filtered_abstract = [w for w in abstract if not w.lower() in stopwords.words('english')]
        join_filtered = ' '.join(str(a) for a in filtered_abstract)
        texts_2.append(join_filtered)

    # Load the pretrained BERT tokenizer.
    print(f"Loading {model_name} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Tokenize all of the sentences and map the tokens to their word IDs
    input_ids = []
    attention_masks = []
    for text in texts_2:
        encoded_dict = tokenizer.encode_plus(
                            text,            
                            add_special_tokens = True,
                            max_length = 256,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                            truncation=True)
    
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return input_ids, attention_masks

def CreateTensorDataset(input_ids, attention_masks, labels):
    """
    Generates a tensor dataset from tokenized data

        Parameters:
            input_ids (lst): List with encoded input_ids
            attention_masks (lst): List with attention masks
            labels (lst): List with labels of data
        
        Returns:
            dataset (tensor): Dataset in tensors
    """

    # Convert to tensors
    input_ids_t = torch.cat(input_ids, dim=0)
    attention_masks_t = torch.cat(attention_masks, dim=0)
    labels_t = torch.tensor(labels, dtype=torch.long)
    print(len(input_ids_t), len(attention_masks_t), len(labels_t))

    dataset = TensorDataset(input_ids_t, attention_masks_t, labels_t)
    return dataset

def DataLoad(train, valid, test, batch_size):
    """
    Loads the tensor dataset into the final form with required batch size

        Parameters:
            train (tensor): Training tensor dataset
            valid (tensor): Validation tensor dataset
            test (tensor): Test tensor dataset
            batch_size (int): Batch size (ideally 16 or 32)
        
        Returns:
            train_dataloader (tensor): Final training dataset
            validation_dataloader (tensor): Final validation dataset
            test_dataloader (tensor): Final test dataset

    """
    # Take training samples in random order
    train_dataloader = DataLoader(
                train,
                sampler = RandomSampler(train),
                batch_size = batch_size
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                valid,
                sampler = SequentialSampler(valid),
                batch_size = batch_size
            )

    # For test the order doesn't matter, so we'll just read them sequentially.
    test_dataloader = DataLoader(
                test,
                sampler = SequentialSampler(test),
                batch_size = batch_size
            )
    return train_dataloader, validation_dataloader, test_dataloader

def PreprocessingSVM(df):
    ############################################
    # Title: A guide to Text Classification(NLP) using SVM and Naive Bayes with Python
    # Author: Gunjit Bedi
    # Date: 9 November 2018
    # Type: Tutorial
    # Source: https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34
    ############################################
    # 1. Removing Blank Spaces
    df['text'].dropna(inplace=True)
    # 2. Changing all text to lowercase
    df['medical_abstract_org'] = df['text']
    df['text'] = [entry.lower() for entry in df['text']]
    # 3. Tokenization-In this each entry in the corpus will be broken into set of words
    df['text']= [word_tokenize(entry) for entry in df['text']]
    # 4. Remove Stop words, Non-Numeric and perfoming Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    for index,entry in enumerate(df['text']):
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
        df.loc[index,'text_final'] = str(Final_words) 
    return df

def FeaturizeSVM(df):
    # TFIDF of data for SVM
    Encoder = LabelEncoder()
    Y = Encoder.fit_transform(df['label'])

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(df['text_final'])
    X_Tfidf = Tfidf_vect.transform(df['text_final'])
    words = TfidfVectorizer.get_feature_names_out(df['text_final'])
    return X_Tfidf, Y, words, Tfidf_vect
