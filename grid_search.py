import time
import datetime
import random

import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup

import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from load_data import CreateData, Tokenization, CreateTensorDataset, DataLoad, PreprocessingSVM
from model import LoadModel, GridSearch

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
torch.cuda.manual_seed_all(17)

url_train = 'https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv'
url_test = 'https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_test.csv'
label_names = ['neoplasms', 'digestive system diseases', 'nervous system diseases', 
               'cardiovascular diseases', 'general pathological conditions']
print('Load data from url ...')
train_text, train_label = CreateData(url_train)
test_text, test_label = CreateData(url_test)
train_data = pd.DataFrame(train_text, train_label).reset_index()
train_data.columns = ['label', 'text']
test_data = pd.DataFrame(test_text, test_label).reset_index()
test_data.columns = ['label', 'text']
torch.cuda.empty_cache()
train_text = train_data['text']
test_text = test_data['text']
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

#model_name = "bert-base-uncased"
model_name = "allenai/scibert_scivocab_uncased"

input_ids_train, attention_masks_train = Tokenization(train_text, model_name)
input_ids_test, attention_masks_test = Tokenization(test_text, model_name)

training_dataset = CreateTensorDataset(input_ids_train, attention_masks_train, train_label)
test_dataset = CreateTensorDataset(input_ids_test, attention_masks_test, test_label)

val_dataset, test_dataset = random_split(test_dataset, [0.5, 0.5])

model = LoadModel(model_name, label_names)
param_grid = dict(batch_size = [16, 32], epochs = [2, 3, 4], lr = [2e-5, 3e-5, 5e-5])

best_model, best_setting, best_acc, results = GridSearch(model_name, label_names, training_dataset, val_dataset, test_dataset, device, param_grid)
torch.save(best_model, "best_model.pt")
with open(r'results_scibert_preprocessed.txt', 'w') as fp:
    for result in results:
        fp.write("%s\n" % result)
    print('Done!')    