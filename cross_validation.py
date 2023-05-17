# https://www.philschmid.de/k-fold-as-cross-validation-with-a-bert-text-classification-example
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from load_data import CreateData, Tokenization, CreateTensorDataset, DataLoad
from evaluation import CrossValBert, CrossValSVM
from model import LoadModel, GridSearch
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import random

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
torch.cuda.manual_seed_all(17)

url_train = 'https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv'
url_test = 'https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_test.csv'
label_names = ['neoplasms', 'digestive system diseases', 'nervous system diseases', 
               'cardiovascular diseases', 'general pathological conditions']

train_text, train_label = CreateData(url_train)
test_text, test_label = CreateData(url_test)


n = 10

train_data = pd.DataFrame(train_text, train_label).reset_index()
train_data.columns = ['label', 'text']
test_data = pd.DataFrame(test_text, test_label).reset_index()
test_data.columns = ['label', 'text']
label_names = ['neoplasms', 'digestive system diseases', 'nervous system diseases', 
               'cardiovascular diseases', 'general pathological conditions']

model_name = "bert-base-uncased"
model_name_sci = "allenai/scibert_scivocab_uncased"
torch.cuda.empty_cache()

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
mean,std = CrossValBert(train_data, test_data, k=10, seed=17, model_name=model_name_sci, label_names = label_names, batch_size=16, epochs = 2, lr = 5e-05, device=device)
with open(r'results_cv_sciBERT_imb.txt', 'w') as fp:
    fp.write(str(mean))
    fp.write(str(std))

# print('Done!')  

# mean_svm, std_svm = CrossValSVM(train_data, k=3, seed=17, kernel='sigmoid', C=1)
# with open(r'results_cv_SVM_imb.txt', 'w') as fp:
#     fp.write(str(mean_svm))
#     fp.write(str(std_svm))