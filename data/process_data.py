import sys
import pandas as pd
import numpy as n
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
        Loads data
    Args: 
        messages_filepath: the filepath for messages csv file
        categories_filepath: the filepath for categories csv file
    Returns: 
        df: messages and categories files merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')

    return df

def clean_data(df):
    """
        Clean the the data by renaming columns and removing duplicates 
    Args: 
        df: The original dataframe
    Returns: 
        df: The cleaned dataframe
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].astype(str).apply(lambda x: x.split('-')[1])
        categories[column] = categories[column].astype(int)
    
    df.drop('categories', axis=1, inplace=True)
    df = df.join(categories)
    df.replace(2, 1, inplace=True)
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """
        Save cleaned dataframe into sqlite database
    Args: 
        df: The cleaned dataframe
        database_filename: name of the database
    Returns: 
        None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_response', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
