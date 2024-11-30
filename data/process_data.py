import sys
import pandas as pd
import sqlite3
import numpy as np


######################################
##### FUNCTIONS FOR LOADING DATA #####
######################################

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
    print('... finished!')
    return data


def load_messages_data(messages_filepath):
    """ Parse the CSV file containing message information.

    Args:
        messages_filepath (str): Path to CSV file containing messages info.

    Returns:
        DataFrame: DataFrame of strings/objects with columns 'messages', 'original' and 'genre', and 'id' as index.
    """
    print(f'Parsing {messages_filepath}.')
    data = pd.read_csv(messages_filepath, sep=',', quotechar='"', index_col='id')
    return data


def load_data(messages_filepath, categories_filepath):
    """ Load the CSV data for messages and categories and merge them together.

    Args:
        messages_filepath (str): File path to the messages CSV file.
        categories_filepath (str): File path to the categories CSV file.

    Returns:
        DataFrame: Merged dataframe of messages and categories.
    """
    categories_df = load_categories_data(categories_filepath=categories_filepath)
    messages_df = load_messages_data(messages_filepath=messages_filepath)
    
    print('Joining categories data with messages data.')
    # both datasets have the same length and the same indexes, so 'how' doesn't need to be specified here
    merged_df = categories_df.merge(messages_df, on='id')
    print('... finished!')
    return merged_df


#######################################
##### FUNCTIONS FOR CLEANING DATA #####
#######################################


def drop_bool_columns_with_constant_value(df):
    """ Identify columns of int data type and drop any that are constant value.
    
    Args:
        df (DataFrame): DataFrame to evaluate.

    Returns:
        DataFrame: DataFrame with constant-value int columns removed.
    """
    int_columns = [k for k, v in df.dtypes.items() if v in [int, np.int64]]
    df_int = df[int_columns]
    unique_int_values = df_int.stack().unique()
    for unique_int_value in unique_int_values:
        to_drop = list(df_int.loc[:, df_int.apply(lambda x: x==unique_int_value).all()].columns)
        if to_drop:
            print(f"Dropping int columns that are all {unique_int_value}: {', '.join(to_drop)}")
            df = df.drop(to_drop, axis=1)
    return df


def clean_data(df, categrorial_columns=['genre', 'related']):
    """ Take the raw merged dataframe and apply cleaning steps.

    Args:
        df (DataFrame): DataFrame to clean.
        categorial_columns (list of str): Columns to create dummy variables from (categorical variables with more than 2 levels).
    Returns:
        DataFrame: Cleaned DataFrame.
    """
    
    df = pd.get_dummies(data=df, columns=categrorial_columns, drop_first=True)
    df = check_and_drop_duplicates(df=df)

    # Check if any categorical/boolean column is all a single value, and drop if so.
    # We do not need to keep any column of constants in this context.
    df = drop_bool_columns_with_constant_value(df)

    return df


#####################################
##### FUNCTIONS FOR SAVING DATA #####
#####################################


def save_data(df, database_filepath, table_name='project2'):
    """ Store DataFrame information into an SQLite3 database.

    Args:
        df (DataFrame): Source data to insert into database.
        database_filepath (str): Filename for the stored SQLite3 database.
        table_name (str, optional): Name of database table to write info into. Defaults to 'project2'.
    """
    print(f'Generating database {database_filepath}.')

    # Create a dict to store field/type information for creating the database table. This will be used to ceate the db table.
    # The 2 string fields - 'message' and 'genre' - will be VARCHAR with length of the largest detected string in each field.
    # Everything else will be INTEGER, and 'id' will be the primary key.
    
    def get_max_string_length_in_column(data):
        return data.apply(lambda x: 0 if pd.isnull(x) else len(x)).max()

    db_types = {
        'id': 'INTEGER PRIMARY KEY',
        'related_1': 'INTEGER',
        'related_2': 'INTEGER',
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
        'message': f'VARCHAR({get_max_string_length_in_column(df.message)})', # length of the longest 'message'
        'original': f'VARCHAR({get_max_string_length_in_column(df.original)})', # length of the longest 'original'
        'genre_news': 'INTEGER',
        'genre_social': 'INTEGER',
    }
    # String to use when creating the table. It looops over the dict's k/v pairs and join each field name and type together.
    create_table_str = "CREATE TABLE data (" + ", ". join([f'{k} {v}' for k, v in db_types.items()]) + " );"

    conn = sqlite3.connect(database_filepath) # Connect to db.
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



        

def main_debug(
        messages_filepath = './disaster_messages.csv',
        categories_filepath = './disaster_categories.csv',
        database_filepath = './project_data.sqlite3',
):
    """_summary_

    Args:
        messages_filepath (str, optional): _description_. Defaults to './disaster_messages.csv'.
        categories_filepath (str, optional): _description_. Defaults to './disaster_categories.csv'.
        database_filepath (str, optional): _description_. Defaults to './project_data.sqlite3'.
    """

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
            .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)
    
    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)
    
    print('Cleaned data saved to database!')


def main():
    """_summary_
    """
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