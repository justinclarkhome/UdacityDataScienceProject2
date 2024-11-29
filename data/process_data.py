import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime as dt
import re
import sqlite3


DATA_DIR = './'
DB_TABLE_NAME = 'project2'

def check_and_drop_duplicates(df):
    """ Check for duplicates in df - including the index value - and drop if found.

    This function resets the index first, as the Pandas duplicate check is only on values.
    We want to include the index in the check (the id, which will be the primary 
    key in the database) as more than 1 observation could have the same characteristics.

    Args:
        df (DataFrame): DataFrame to check for duplicates/

    Returns:
        DataFrame: DataFrame with dupes dropped (if any).
    """
    if df.reset_index().duplicated().any():
        print('... duplicates detected: they will be dropped.')
        df = df.reset_index().drop_duplicates().set_index('id')
    return df


def load_categories_data(categories_filepath, skip_rows=1):
    """ Parse the CSV file containing category information.

    Args:
        categories_filepath (str): Path to CSV file containing category info.
        skip_rows (int, optional): Number of rows to skip when parsing (e.g. for the header). Defaults to 1.

    Returns:
        DataFrame: DataFrame of ints (0 or 1) with categories as columns and 'id' as index.
    """
    data = {}
    print(f'Parsing {categories_filepath}.')
    with open(categories_filepath, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i > skip_rows - 1: # Remember: Python indexes are 0-based!
                this_line_info = {} # Holder for just this line's info.
                identifier, keyvalue = line.split(',')
                for item in keyvalue.split(';'):
                    k, v = item.strip().split('-')
                    this_line_info[k] = v
                data[int(identifier)] = this_line_info

    # Convert to dtype int and name the index (so we can join on it later).
    data = pd.DataFrame(data).T.rename_axis('id', axis=0).astype(int)
    data = check_and_drop_duplicates(df=data)
    print('... finished!')
    return data


def load_messages_data(messages_filepath, skip_rows=1):
    """ Parse the CSV file containing message information.

    Each line has a vaguely consistent format, something like:
    - 'id', 'message' (in english), 'original' (message in original language), 'genre'.
    - 'id' is always at the beginning, always a number, and always followed by a comma.
    - 'genre' is always at the end, and always preceded by a comma.
    - The rest is harder to parse, because 'original' is sometimes blank (in between two commas, followed by 'genre').
    - Sometimes 'message' is surrounded in quotes. It LOOKs like it gets surrounded in quotes if it contains a comma.

    While it is difficult to separate 'message' and 'original' when both are present, in the ETL stage we will
    leave both as one string. Then in the ML pipeline we can try to clean the string with NLTK, to strip
    out non-English words as best as possible.

    Args:
        messages_filepath (str): Path to CSV file containing messages info.
        skip_rows (int, optional): Number of rows to skip when parsing (e.g. for the header). Defaults to 1.

    Returns:
        DataFrame: DataFrame of strings with columns 'message' and 'genre' and 'id' as index.
    """
    data = {}
    print(f'Parsing {messages_filepath}.')
    with open(messages_filepath, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i > skip_rows - 1: # Remember: Python indexes are 0-based!
                line = line.strip() # Get rid any leading/trailing whitespace, line endings, etc.

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
    data = check_and_drop_duplicates(df=data)
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

    # for removing non-English words:
    # https://stackoverflow.com/questions/41290028/removing-non-english-words-from-text-using-python
    pass


def save_data(df, database_filename, table_name=DB_TABLE_NAME):
    """ Store DataFrame information into an SQLite3 database.

    Args:
        df (DataFrame): Source data to insert into database.
        database_filename (str): Filename for the stored SQLite3 database.
        table_name (str, optional): Name of database table to write info into. Defaults to DB_TABLE_NAME.
    """
    print(f'Generating database {database_filename}.')

    # Create a dict to store field/type information for creating the database table. This will be used to ceate the db table.
    # The 2 string fields - 'message' and 'genre' - will be VARCHAR with length of the largest detected string in each field.
    # Everything else will be INTEGER, and 'id' will be the primary key.
    db_types = {
        'id': 'INTEGER PRIMARY KEY',
        'related': 'INTEGER',
        'request': 'INTEGER',
        'offer': 'INTEGER',
        'aid_related': 'INTEGER',
        'medical_help': 'INTEGER',
        'medical_products': 'INTEGER',
        'search_and_rescue': 'INTEGER',
        'security': 'INTEGER',
        'military': 'INTEGER',
        'child_alone': 'INTEGER',
        'water': 'INTEGER',
        'food': 'INTEGER',
        'shelter': 'INTEGER',
        'clothing': 'INTEGER',
        'money': 'INTEGER',
        'missing_people': 'INTEGER',
        'refugees': 'INTEGER',
        'death': 'INTEGER',
        'other_aid': 'INTEGER',
        'infrastructure_related': 'INTEGER',
        'transport': 'INTEGER',
        'buildings': 'INTEGER',
        'electricity': 'INTEGER',
        'tools': 'INTEGER',
        'hospitals': 'INTEGER',
        'shops': 'INTEGER',
        'aid_centers': 'INTEGER',
        'other_infrastructure': 'INTEGER',
        'weather_related': 'INTEGER',
        'floods': 'INTEGER',
        'storm': 'INTEGER',
        'fire': 'INTEGER',
        'earthquake': 'INTEGER',
        'cold': 'INTEGER',
        'other_weather': 'INTEGER',
        'direct_report': 'INTEGER',
        'message': f'VARCHAR({df.message.apply(lambda x: len(x)).max()})', # length of the longest 'message'
        'genre': f'VARCHAR({df.genre.apply(lambda x: len(x)).max()})', # length of the longest 'genre'
    }
    # String to use when creating the table. It looops over the dict's k/v pairs and join each field name and type together.
    create_table_str = "CREATE TABLE data (" + ", ". join([f'{k} {v}' for k, v in db_types.items()]) + " );"

    conn = sqlite3.connect(database_filename) # Connect to db.
    cur = conn.cursor() # Get a cursor.

    cur.execute("DROP TABLE IF EXISTS data") # Drop the 'data' table (in case we're re-writing it).
    conn.commit()

    print(f'... creating table "{table_name}"')
    cur.execute(create_table_str) # Create the 'data' table.

    print(f'... inserting data into "{table_name}" from DataFrame.')
    df.to_sql(name=table_name, con=conn, if_exists='replace') # Insert data from df into the database.
    conn.commit()

    conn.close()
    print('... finished!')
    

def main(
        messages_filepath = os.path.join(DATA_DIR, 'disaster_messages.csv'),
        categories_filepath = os.path.join(DATA_DIR, 'disaster_categories.csv'),
        database_filepath = os.path.join(DATA_DIR, 'project_data.sqlite3'),
):
    """_summary_

    Args:
        messages_filepath (_type_, optional): _description_. Defaults to os.path.join(DATA_DIR, 'disaster_messages.csv').
        categories_filepath (_type_, optional): _description_. Defaults to os.path.join(DATA_DIR, 'disaster_categories.csv').
        database_filepath (_type_, optional): _description_. Defaults to os.path.join(DATA_DIR, 'project_data.sqlite').
    """

    # Load raw source data and combine.
    df = load_data(messages_filepath, categories_filepath)
    
    # Clean combined data.

    # Save cleaned data to on-disk database.
    save_data(df, database_filepath)


def main_example():
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