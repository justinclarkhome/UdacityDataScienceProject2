import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime as dt
import re


DATA_DIR = './'

def load_categories_data(categories_filepath, skip_rows=1):
    """ Parse the CSV file containing category information.

    Args:
        categories_filepath (str): Path to CSV file containing category info.
        skip_rows (int, optional): Number of rows to skip when parsing (e.g. for the header). Defaults to 1.

    Returns:
        _type_: _description_
    """
    data = {}
    print(f'Parsing {categories_filepath}.')
    with open(categories_filepath, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i > skip_rows - 1: # remember: Python indexes are 0-based
                this_line_info = {} # holder for just this line's info
                identifier, keyvalue = line.split(',')
                for item in keyvalue.split(';'):
                    k, v = item.strip().split('-')
                    this_line_info[k] = v
                data[int(identifier)] = this_line_info
    data = pd.DataFrame(data).T.rename_axis('id', axis=0) # name the index, so we can join on it later
    print('... finished!')
    return data


def load_messages_data(messages_filepath, skip_rows=1):
    """_summary_

    Args:
        messages_filepath (_type_): _description_
        skip_rows (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    data = {}
    print(f'Parsing {messages_filepath}.')
    with open(messages_filepath, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i > skip_rows - 1: # remember: Python indexes are 0-based
                line = line.strip() # get rid any leading/trailing whitespace, line endings, etc

                # each line has a vaguely consistent format, something like
                # id, message (in english), original (message in original language), genre
                # id is always at the beginning, always a number, and always followed by a comma
                # genre is always as the end, and always preceded by a comma
                # the rest is harder, because the original is sometimes blank (in between two commas, followed by the genre)
                # sometimes message is surrounded in quotes, sometimes not. It LOOKs like it gets surrounded in quotes if it contains a comma.
                # so we can try to apply a regex looking pairs of quotes with a comma in middle
                # we can also look for two consecutive commas followed by the genre to isolate instances where the original is blank

                # I'm not sure there is a way to split out the english/non-english sections. And maybe it is BETTER not to try, in case there is extra predictive content in there?

                # the idenififier is always a number, always at the beginning, and always followed by a comma
                identifier = re.search(r'^[0-9]+,', line).group() # .replace(',', '')
                
                # split the string on the idenifier - it'll be at the beginning. Then isolate the second part of the split.
                remaining = line.split(identifier)[-1]
                
                # the genre (last segment) is always a single word at the end of the string, preceded by a comma
                genre = re.search(r',[a-zA-Z]+$', line).group()

                # split the string on the genre - it'll be at the endg. Then isolate the first part of the split.
                remaining = remaining.split(genre)[0]

                data[int(identifier.strip(','))] = {
                    'message': remaining.replace(',', '').strip('"'),
                    'genre': genre.strip(','),
                }

                # no original - e.g. the string ends with two commas and the genre
                # no_original_check = re.search(f",{genre}$", line)
                # if no_original_check:
                #     original = None
                #     message = remaining.strip('"')

    data = pd.DataFrame.from_dict(data).T.rename_axis('id', axis=0) # name the index, so we can join on it later
    print('... finished!')
    return data


def load_data(messages_filepath, categories_filepath):
    """_summary_

    Args:
        messages_filepath (_type_): _description_
        categories_filepath (_type_): _description_

    Returns:
        _type_: _description_
    """
    categories_df = load_categories_data(categories_filepath=categories_filepath)
    messages_df = load_messages_data(messages_filepath=messages_filepath)
    
    print('Joining categories data with messages data.')
    # both datasets have the same length and the same indexes, so 'how' doesn't need to be specified here
    merged_df = categories_df.join(messages_df, on='id')
    print('... finished!')
    return merged_df


def clean_data(df):
    pass


def save_data(df, database_filename):
    pass  


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