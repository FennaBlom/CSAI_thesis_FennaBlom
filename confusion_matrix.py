from model import TrainModel, EvaluationModel, LoadModel
from load_data import CreateData, Tokenization, CreateTensorDataset, DataLoad, PreprocessingSVM, FeaturizeSVM
import pickle
from evaluation import ConfusionMatrix
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import TensorDataset, random_split
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import numpy as np
from sklearn import model_selection, naive_bayes, svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


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

# data bert

model_name = "bert-base-uncased"

input_ids_train, attention_masks_train = Tokenization(train_text, model_name)
input_ids_test, attention_masks_test = Tokenization(test_text, model_name)

training_dataset = CreateTensorDataset(input_ids_train, attention_masks_train, train_label)
test_dataset = CreateTensorDataset(input_ids_test, attention_masks_test, test_label)

val_dataset, test_dataset = random_split(test_dataset, [0.5, 0.5])

train_dataloader, validation_dataloader, test_dataloader = DataLoad(training_dataset, val_dataset, test_dataset, 32)

# load model bert
model_bert = torch.load('best_model.pt')

# predictions bert
model_scibert = torch.load('model_scibert.pt')
model_scibert.cuda()

true_labels_bert, predictions_bert, acc_bert, f1_bert = EvaluationModel(model_bert, test_dataloader, device)

labels = {0: "neoplasms", 1: "digestive system diseases", 2: "nervous system diseases",
          3: "cardiovascular diseases", 4: "general pathological conditions"}

true_bert = [labels[x] for x in true_labels_bert]
pred_bert = [labels[x] for x in predictions_bert]

# confusion matrix bert
cm_bert = ConfusionMatrix(true_bert, pred_bert, labels=label_names)
cm_bert.plot(cmap=plt.cm.Blues, xticks_rotation=45)

# data scibert
model_name = 'allenai/scibert_scivocab_uncased'
input_ids_train_scibert, attention_masks_train_scibert = Tokenization(train_text, model_name)
input_ids_test_scibert, attention_masks_test_scibert = Tokenization(test_text, model_name)

training_dataset_scibert = CreateTensorDataset(input_ids_train_scibert, attention_masks_train_scibert, train_label)
test_dataset_scibert = CreateTensorDataset(input_ids_test_scibert, attention_masks_test_scibert, test_label)

val_dataset_scibert, test_dataset_scibert = random_split(test_dataset_scibert, [0.5, 0.5])

train_dataloader_scibert, validation_dataloader_scibert, test_dataloader_scibert = DataLoad(training_dataset_scibert, val_dataset_scibert, test_dataset_scibert, 32)

# predictions scibert
true_labels_scibert, predictions_scibert, acc_scibert, f1_scibert = EvaluationModel(model_scibert, test_dataloader_scibert, device)

true_scibert = [labels[x] for x in true_labels_scibert]
pred_scibert = [labels[x] for x in predictions_scibert]

# confusion matrix scibert
cm_scibert = ConfusionMatrix(true_scibert, pred_scibert, labels=label_names)
cm_scibert.plot(cmap=plt.cm.Blues, xticks_rotation=45)

#svm
df_train = PreprocessingSVM(train_data)
df_train['text_final'] = df_train['text_final'].astype(str)
Train_X_Tfidf, Train_Y = FeaturizeSVM(df_train)

SVM = svm.SVC(kernel='sigmoid', C=1.0, degree=3, gamma=1,random_state=17)
test_data = pd.DataFrame(test_text, test_label).reset_index()
test_data.columns = ['label', 'text']
Encoder = LabelEncoder()
df_test = PreprocessingSVM(test_data)

df_test['text_final'] = df_test['text_final'].astype(str)
val_X_svm, test_X_svm = random_split(df_test, [0.5, 0.5])
test_X_svm = test_X_svm.dataset.iloc[test_X_svm.indices]

Test_Y = Encoder.fit_transform(test_X_svm['label'])
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df_train['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(df_train['text_final'])
Test_X_Tfidf = Tfidf_vect.transform(test_X_svm['text_final'])
SVM.fit(Train_X_Tfidf, Train_Y)

# predictions svm
predictions_svm = SVM.predict(Test_X_Tfidf)
true_labels_svm = Test_Y
true_svm = [labels[x] for x in true_labels_svm]
pred_svm = [labels[x] for x in predictions_svm]


# confusion matrix svm
cm_svm = ConfusionMatrix(true_svm, pred_svm, label_names)
cm_svm.plot(cmap=plt.cm.Blues, xticks_rotation=45)

# combine all matrices
classifiers = {"SVM": cm_svm, "BERT": cm_bert, "SciBERT": cm_scibert}
f, axes = plt.subplots(1, 3, figsize=(20,5), sharey='row')

for i, (key, cm) in enumerate(classifiers.items()):
    disp = cm
    disp.plot(ax=axes[i], xticks_rotation=45)
    disp.ax_.set_title(key)
    disp.im_.colorbar.remove()
    if i!=1:
        disp.ax_.set_xlabel('')
    if i!=0:
        disp.ax_.set_ylabel('')

plt.subplots_adjust(wspace=0.40, hspace=0.1)
f.colorbar(disp.im_, ax=axes)

plt.savefig('confusion_matrices.png',bbox_inches = 'tight')
plt.show()
