"""
Ross Heaton github.com/rossheat

The purpose of this scrit is to train an NLP model from data stored inside of a SQL database.

Example usage:

>>> python train_classifier.py <sql-database-path> <pickle-file-output-path> 

"""

# Matrix math, constants, types
import numpy as np
# Data wranggling and cleaning 
import pandas as pd
# Commandline args 
import sys
# Natural language processing tools 
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
# Macine learning and processing tools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import fbeta_score, make_scorer
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
# Regular expressions 
import re
# Operating system paths
import os 
# Save pickle files 
import pickle
# Fetch data from SQL database
from sqlalchemy import create_engine

def load_data(database_filepath):
    
    """
    Loads the data from the SQL database into memory

    Args:
        1. database_filepath - filepath to the database
    Returns:
        1. X - the training features
        2. y - the training targets 
        3. category_names - list of category names 
    """
    
    # Create the SQL engine using the filepath to the database
    engine = create_engine("sqlite:///" + database_filepath)
    # Load the data into a DataFrame from the database table 
    df = pd.read_sql_table("DisasterResponse", engine)
    
    # All rows from the message column
    X = df.loc[:, 'message']
    # All rows from the 5th column onwards
    y = df.iloc[:, 4:]
    
    # Get the category names from the target DataFrame column names
    category_names = list(y.columns)
    
    return X, y, category_names


def tokenize(text):
    """
    Tokenizzes and cleans text
    
    Args:
        1. text - message to be tokenized
    Returns:
        1. clean_tokens - List of tokenized text
    """
    
    # Get token from text
    tokens = word_tokenize(text)
    # Init lemmatizer object 
    lemmatizer = WordNetLemmatizer()
    # Init empty list that will eventually contain clean tokens
    clean_tokens = []
    # Loop through each token
    for tok in tokens:
        # Lemmatize, strip, and lower case the token
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        # Append the cleaned token to the clean_tokens list 
        clean_tokens.append(clean_tok)

    # Return the list of cleaned tokens 
    return clean_tokens

def build_model(X_train, Y_train):
    
    """
    Builds the model using a Pipeline 
    
    Args:
        1. X_train - the training features 
        2. Y_train - the training labels
    Returns:
        1. cv - a model trained with GridSearch
    """
    
    # Create and configure the Pipeline object 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # The parameters for the grid search 
    params = {  
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__max_features': [None, 'log2', 'auto', 'sqrt'],
        'clf__estimator__max_depth': [None, 10, 50, 100],
    }
    
    # Init the grid search object with the pipeline and grid search parameters 
    cv = GridSearchCV(estimator=pipeline, param_grid=params)
    # Train the grid search object using the training features and labels
    cv.fit(X_train,Y_train)
    # Return the trained model 
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the trained model using an unseen dataset via a Classification Report
    
    Args: 
        1. model - the trained model to be evaluated 
        2. X_test - the test features
        3. Y_test - the test labels 
        4. category_names - a list of category names 
        
    Returns: 
        None
    """
    
    # Use the trained model to predict the test data sets target variables
    Y_pred = model.predict(X_test)
    # Loop over each category label and print a classification report 
    for idx, col in enumerate(category_names):
        print(col, classification_report(Y_test.iloc[:,idx], Y_pred[:,idx]))


def save_model(model, model_filepath):
    """
    Saves the trained model to a pickle file
    
    Args:
        1. model - the model we wish to save 
        2. model_filepath - the path to which we wish to save the model
    
    Returns:
        None 
    """
    # Save the model in a pickle file under the key 'model'
    pickle.dump({'model': model}, open(model_filepath, "wb"))


def main():
    
    # If the user has entered the correct number of commandline arguments, continue with normal execution, otherwise, prompt the user to input the correct number/type of commandline arguments 
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()