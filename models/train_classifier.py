import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])

import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
        Loads cleaned data from the sqlite database
    Args: 
        database_filepath: path of the database file
    Returns: 
        X: messages dataframe (target variable)
        Y: features dataframe (features)
        category_names: list of column names
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_response', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    return X, Y, Y.columns


def tokenize(text):
    """
      tokenize text
    Args: 
        text: the text to tokenize
    Returns: 
        cleaned tokens
    """   
    url_re = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_re, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """
      build model pipeline and run hyperparameter tuning
    Args: 
        None
    Returns: 
        cross validated classifier object
    """   
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'clf__estimator__n_estimators': [50],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
        Evaluate the model performances using classification report
    Args: 
        model: the model to be evaluated
        X_test: X_test dataframe
        Y_test: Y_test dataframe
        category_names: list of category names
    Returns: 
        None
    """ 
        
    y_pred = model.predict(X_test)

    for columns in Y_test:
        print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
        Save model to pickle
    Args: 
        model: the model to be saved
        model_filepath: filepath to be saved in
    Returns: 
        None
        
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
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
