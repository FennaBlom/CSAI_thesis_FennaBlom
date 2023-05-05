import numpy as np
import time
import datetime
import random
import itertools


from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score


import torch
from load_data import DataLoad


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def LoadModel(model_name, label_names):
    """
    Loads and returns a pre-trained model
    
        Parameters:
            model_name (str): Name or location of pre-trained model
            label_names (lst): List with all categories (str)
        
        Returns:
            model: Loaded pre-trained model
    """

    # Load pretrained model for sequence classification
    print(f"Loading {model_name} model...")
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(label_names)
    config.output_attentions = True
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config)
    return model 



def TrainModel(train_data, val_data, model, epochs, lr, device):
    """
    Trains and returns a model

        Parameters:
            train_data (tensor): Training tensor dataset
            val_data (tensor): Validation tensor dataset
            model (?): Loaded pre-trained model
            epochs (int): Number of epochs (2, 3, or 4)
            lr (float?): Learning rate
            device (str): Whether a GPU or CPU should be used

        Returns:
            model: Trained model
    """
    optimizer = AdamW(model.parameters(),
                  lr = lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_data) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_data):

            # Progress update every 40 batches.
            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_data), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            loss, logits, attentions = model(input_ids=b_input_ids, 
                                            attention_mask=b_input_mask, 
                                            labels=b_labels).to_tuple()

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_data)            
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in val_data:
            
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                loss, logits, attentions = model(input_ids=b_input_ids, 
                                                attention_mask=b_input_mask,
                                                labels=b_labels).to_tuple()
                
                
            # Accumulate the validation loss.

            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(val_data)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_data)
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    return model

def EvaluationModel(model, test_data, device):
    """
    Evaluates the model on the test set and returns the predictions and accuracy

        Parameters:
            model (?): Trained model
            test_data (tensor): Final test dataset
            device (str): Whether GPU or CPU should be used

        Returns:
            predictions (array): Predicted classes for the test set 
            accuracy (float): The accuracy on the test set
    """
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    print("Predicting labels ...")
    # Predict 
    for batch in test_data:
        batch = tuple(t.to(device) for t in batch)
    
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(input_ids=b_input_ids,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
    
        # Store predictions and true labels
        predictions.extend(logits)
        true_labels.extend(label_ids)

    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(true_labels, predictions)
    return true_labels, predictions, accuracy

def GridSearch(model_name, label_names, train_data, val_data, test_data, device, params_grid):
    """
    Explores the hyperparameter space and returns the best performing hyperparameter combination

        Parameters:
            model_name: Name of the pre-trained model
            train_data (tensor): Training dataset
            val_data (tensor): Validation dataset
            test_data (tensor): Test dataset
            device (str): Whether GPU or CPU should be used
            params_grid (dict): All possible combinations of the hyperparameters

        Returns:
            best_model: The best performing model
            best_setting (dict): Dictionary with the best hyperparameter settings
            accuracy (float): The accuracy of the best model on the test dataset
            results (list): List with the performance of all hyperparameter settings

    """
    model = LoadModel(model_name, label_names)
    keys, values = zip(*params_grid.items())
    param_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_setting = param_dicts[0]
    best_acc = 0
    best_model = model
    results = []
    setting_count = 0
    for params in param_dicts:
        model = LoadModel(model_name, label_names)
        model.cuda()
        batch_size = params['batch_size']
        epochs = params['epochs']
        lr = params['lr']
        train_dataloader, validation_dataloader, test_dataloader = DataLoad(train_data, val_data, test_data, batch_size)
        print('======== Setting {:} / {:} ========'.format(setting_count + 1, len(param_dicts)))
        setting_count += 1
        new_model = TrainModel(train_dataloader, validation_dataloader, model, epochs, lr, device)
        test_pred, test_acc = EvaluationModel(new_model, test_dataloader, device)
        print(test_acc)
        results.append([params, test_acc])
        if test_acc > best_acc:
            best_acc = test_acc
            best_setting = params
            best_model = new_model
            print("The current model is better")
    return best_model, best_setting, best_acc, results
