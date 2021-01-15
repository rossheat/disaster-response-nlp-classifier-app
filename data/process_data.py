"""
Ross Heaton github.com/rossheat

The purpose of this script is to preprocess data before loading it into an SQL database. 

Example usage: 
>>> python process_data.py <disaster-messages-csv-file-path> <disaster-categories-csv-file-path> <sqllite-database-path>

"""

# Commandline args
import sys
# Matrix math, constants, and types
import numpy as np
# Data wranggling and cleaning 
import pandas as pd
# Loading cleaned data to SQL database 
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """
    Loads the disaster messages CSV and disaster categories CSV files into DataFrame's and merges the two files into one data frame on the 'id' variable. 
    
    Args:
        1. messages_filepath - path to the CSV file containing disaster messages. 
        2. categories_filepath - path to the CSV file containing disaster categories.
    Returns:
        1. df - DataFrame containg disaster messages and categories.  

    """
    
    # Load the messages CSV into a DataFrame 
    messages = pd.read_csv(messages_filepath)
    # Load the categories CSV into a DataFrame
    categories = pd.read_csv(categories_filepath)
    # Merge the DataFrames on the 'id' column 
    df = pd.merge(messages,categories, on='id')
    # Return the merged DataFrame ready to be passed to the clean_data function
    return df 


def clean_data(df):
    
    """
    Cleans the DataFrame returned by the load_data function ready for it to be loaded into a SQL database.
    
    Args:
        1. df - DataFrame containg disaster messages and categories.
    Returns: 
        1. df - Cleaned version of the DataFrame passed as an argument to the function.
    """
     
    # split the Strings within the 'categories' columns on the ';' char, and expand out the resulting values into a DataFrame called categories
    categories = df['categories'].str.split(pat=';', expand=True)
    # Obtain a row from the categories DataFrame and perform cleaning 
    row = categories.iloc[[1]]
    # Get new column names 
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    # Assign new column names to the categories DataFrame
    categories.columns = category_colnames
    # Loop through each column in categories DataFrame
    for column in categories:
        # Last char in String
        categories[column] = categories[column].str[-1]
        # Convert the values to type int
        categories[column] = categories[column].astype(np.int)
    
    # Remove the 'categories' column from the df DataFrame
    df = df.drop('categories', axis='columns')
    
    # Concatinate the df DataFrame and the Categories DataFrame
    df = pd.concat([df,categories],axis='columns')
    
    # Drop any duplicate records from the df DataFrame 
    df = df.drop_duplicates()
    
    # Returned the cleaned DataFrame ready to be passed to the save_data function 
    return df


def save_data(df, database_filename):
    
    """
    Saves the cleaned DataFrame returned from the clean_data function to a SQL (sqllite) database. 
    
    Args:
        1. df - The cleaned DataFrame.
        2. database_filename - database file name.
    Returns:
        None
    """
    
    # Create a new database engine
    engine = create_engine('sqlite:///'+ database_filename)
    # Save the df as a SQL database using the table name and engine 
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    
    # If the user has not specified the correct number of command-line arguments, prompt them to do so. If they have, extract the arguments and continute with the ETL process. 
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

# Only run this module when it executed explicitly - not when it is imported 
if __name__ == '__main__':
    main()