# Building a Conversational Agent with Rasa to Enrich a Medical Abstracts Dataset
This README file provides an overview of the thesis project and instructions on how to understand and use the code.

# Introduction
The goal of this thesis project was to build a conversational agent using Rasa that enriches a medical abstracts dataset by allowing users to add new papers to the dataset. The new papers were classified using topic classification, and three methods—SVM, BERT, and SciBERT—were considered for topic classification. The code for the topic classification models is included in this repository. However, the code for the conversational agent itself is not included. The code for the conversational agent can be found here: https://github.com/FennaBlom/CSAI_thesis_FennaBlom-rasa


# Prerequisites
Before using the code, make sure you have the following prerequisites installed:

NLTK version 3.8.1
scikit-learn 1.0.2
Huggingface Transformers 4.26.1
PyTorch 1.13.1
SHAP 0.41.0
Rasa version 3.5.4

# Installation
To install the required dependencies, you can use the following command:

shell
pip install -r requirements.txt


# Usage
The code can be used by running different Python files from the command line. Here's an overview of the available files and their purposes:

grid_search.py: Performs grid search for BERT and SciBERT models. To train the correct model, uncomment either line 64 or 65. The best model will be saved as a .pt file, and the results will also be saved in a .txt file.
svm.py: Performs grid search for SVM. Uncomment lines 104, 105, and 106 to perform the grid search.
cross_validation.py: Runs 10-fold cross-validation. The model name might need to be adjusted in the code for the corresponding model.
confusion_matrix.py: Creates confusion matrices. The confusion matrix will be automatically saved as a .png file.
SHAP: SHAP computations can only be done in the SHAP notebook.
For command line usage, no specific arguments are required.

# Known Issues
Results of SVM for cross-validation and confusion matrix might not match.
The repository does not include the trained models as they were too big to upload. These models, however, are available upon request.

# Contributing
As this is a thesis project, contributions are not expected. However, if you have any suggestions or improvements, feel free to reach out to the project owner.

# Contact
For any questions or inquiries about this project, please contact the project owner at fennablom@hotmail.com.


