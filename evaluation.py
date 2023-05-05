from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from load_data import Tokenization, CreateTensorDataset, DataLoad, PreprocessingSVM, FeaturizeSVM
from model import TrainModel, EvaluationModel, LoadModel
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import numpy as np
from sklearn import model_selection, naive_bayes, svm


def CrossValBert(train_data, test_data, model_name, label_names, batch_size, epochs, lr, device, k, seed):
    """
    Performs a k-fold cross-validation of BERT

        Parameters:
            train_data (df): Dataframe of the training data
            test_data (df): Dataframe of the test data
            model_name: Name or location of pre-trained model
            label_names (lst): List with all categories (str)
            batch_size (int): Batch size for BERT
            epochs (int): Number of epochs (2, 3, or 4)
            lr (float): Learning rate
            device (str): Whether a GPU or CPU should be used
            k (int): Number of folds
            seed (int): Setting the seed
        
        Returns:
            accuracy (lst): List of k accuracy scores
            f1 (lst): List of k F1-scores
        
    """
    kf = KFold(n_splits=k, random_state=seed, shuffle=True)
    accuracy = []
    f1 = []
    for train_index, val_index in kf.split(train_data):
        # split Dataframe
        train_df = train_data.iloc[train_index]
        val_df = train_data.iloc[val_index]
        # load data into correct format
        input_ids_train, attention_masks_train = Tokenization(train_df['text'], model_name)
        input_ids_val, attention_masks_val = Tokenization(val_df['text'], model_name)
        input_ids_test, attention_masks_test = Tokenization(test_data['text'], model_name)
        train_label = train_df['label'].to_numpy()
        test_label = test_data['label'].to_numpy()
        val_label = val_df['label'].to_numpy()       

        train_dataset = CreateTensorDataset(input_ids_train, attention_masks_train, train_label)
        val_dataset = CreateTensorDataset(input_ids_val, attention_masks_val, val_label)
        test_dataset = CreateTensorDataset(input_ids_test, attention_masks_test, test_label)
        train_dataloader, validation_dataloader, test_dataloader = DataLoad(train_dataset, val_dataset, test_dataset, batch_size)
        # load model
        model = LoadModel(model_name, label_names)
        model.cuda()
        # train model
        model = TrainModel(train_dataloader, validation_dataloader, model, epochs, lr, device)
        # test model
        test_pred, test_acc, f1_weighted = EvaluationModel(model, test_dataloader, device)
        # append model score
        accuracy.append(test_acc)
        f1.append(f1_weighted)
    return accuracy, f1

def CrossValSVM(train_data,  k, seed, kernel, C):
    """
    Performs a k-fold cross-validation of SVM

        Parameters:
            train_data (df): Dataframe of the training data
            k (int): Number of folds
            seed (int): Sets the seed
            kernel (str): Type of kernel
            C (float): Regularization parameter
    
        Returns:
            accuracy (lst): List of k accuracy scores
            f1 (lst): List of k F1-scores
    """
    results = []
    SVM = svm.SVC(kernel=kernel, C=C, degree=3, gamma=1,random_state=seed)
    # Perform cross-validation
    df_train = PreprocessingSVM(train_data)
    df_train['text_final'] = df_train['text_final'].astype(str)
    Train_X_Tfidf, Train_Y = FeaturizeSVM(df_train)
    accuracy = cross_val_score(SVM, Train_X_Tfidf, Train_Y, cv=k, scoring='accuracy')
    f1_weighted = cross_val_score(SVM, Train_X_Tfidf, Train_Y, cv=k, scoring=make_scorer(f1_score, average='weighted'))
    return accuracy, f1_weighted
        
def ConfusionMatrix(y_true, y_predict, labels):
    """"
    Creates the confusion matrix

        Parameters:
            y_true (lst): List of true y values
            y_predict (lst): List of predicted y values
            labels (lst): List of all labels
        
        Returns:
            disp: Display of the confusion matrix
    """
    cm = confusion_matrix(y_true, y_predict, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    return disp