import sys
import os
import pandas as pd
import re
import sqlite3
import nltk


DATA_DIR = './'
DB_TABLE_NAME = 'project2'

# Necessary NLTK downloads (for filtering English words)
nltk.download('words');
nltk.download('wordnet');

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
    data = check_and_drop_duplicates(df=data)
    print('... finished!')
    return data


def load_messages_data(messages_filepath):
    data = pd.read_csv(messages_filepath, sep=',', quotechar='"', index_col='id')
    data = check_and_drop_duplicates(df=data)
    return data


def load_messages_data_OLD(messages_filepath, skip_rows=1):
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
                    'message_raw': remaining, # .replace(',', '').strip('"'),
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


#######################################
##### FUNCTIONS FOR CLEANING DATA #####
#######################################

def clean_data(df):
    """_summary_

    Args:
        df (DataFrame): DataFrame to clean.

    Returns:
        DataFrame: Cleaned DataFrame.
    """

    # Pull out English component of raw message, if possible.
    # df, _ = extract_english_content_from_raw_messages(df=df, tolerance=0.35)

    # I think this should be part of the ML pipeline.
    # df = pd.get_dummies(data=df, columns=['genre'], drop_first=True)

    return df


def get_english_words_in_string(s, english_word_set=set(nltk.corpus.words.words()), adhoc_words=()):
    """_summary_

    Args:
        s (str): String to parse.
        english_words (set, optional): A set of known English words. Defaults to set(nltk.corpus.words.words()).
        adhoc_words (collection, optional): Additional words to consider as English, along with english_word_set.

    Returns:
        tuple: Three-tuple containing ths set of all words in the string, the set of all English words in the string, and the ratio.
    """

    all_words = [i for i in nltk.wordpunct_tokenize(s.lower()) if i.isalpha()]
    english_words = [i for i in all_words if i in english_word_set or i in adhoc_words]
    if all_words:
        ratio_of_english_words = len(set(english_words))/len(set(all_words))
    else:
        ratio_of_english_words = None
    return all_words, english_words, ratio_of_english_words


# apply this in the cleaning phase to see if there's any point?
def extract_english_content_from_raw_messages(df, raw_message_field='message_raw', english_message_field='message_english', tolerance=0.35, adhoc_words=()):
    """ Parse an input message string, attempting to identify the English segment of it, based on punctuation present in the string 
    or the ratio of English words in it being larger than tolerance.

    Thank you: https://stackoverflow.com/questions/41290028/removing-non-english-words-from-text-using-python

    Args:
        df (_type_): In DataFrame containing messages.
        message_field (str, optional): Column of DataFrame containing the messages to parse. Defaults to 'message_raw'.
        tolerance (float, optional): Ratio of English words required to define a segment of the string as English. Defaults to 0.35.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    inspect = {} # For debugging.
    cleaned_english_tokens = {}
    likely_english_messages_raw = {}

    # Loop over raw messages, evaluate each based on the content of the string.
    for k, v in df[raw_message_field].items():
        # Check if string contains a quotation mark - if so, we know it contains a 'message' segment and an 'original' segment.
        # we can loop over and attempt to identify which is which
        if v.count('"') > 0:
            english_tokens_in_this_message = []
            english_part_of_this_message = []
            segments = v.split('"')
            for segment_raw in segments:
                if len(segment_raw) > 0:
                    all_words, english_words, ratio_of_english_words = get_english_words_in_string(
                        s=segment_raw,
                        adhoc_words=adhoc_words,
                        )

                    if all_words:
                        # If the ratio is more than the tolerance, consider this segment as English and add it to clean_message.
                        if ratio_of_english_words > tolerance:
                            english_part_of_this_message.append(segment_raw)
                            english_tokens_in_this_message += english_words
                        else:
                            if k in inspect:
                                inspect[k].append(segment_raw)
                            else:
                                inspect[k] = [segment_raw]
                if english_tokens_in_this_message:
                    cleaned_english_tokens[k] = list(set(english_tokens_in_this_message))
                    likely_english_messages_raw[k] = ' '.join(english_part_of_this_message)
        elif v.count(',') > 0:

            # If no quotation mark but there IS a comma, then the first part of the CSV string should be the English message.
            # But check to make sure it has more English words than the tolerance level.
            english_part_of_this_message = v.split(',')[0]
            all_words, english_words, ratio_of_english_words = get_english_words_in_string(english_part_of_this_message)
            if ratio_of_english_words and ratio_of_english_words > tolerance:
                likely_english_messages_raw[k] = english_part_of_this_message
            else:
                if k in inspect:
                    inspect[k].append(v)
                else:
                    inspect[k] = [v]
        else:
            raise ValueError('Unexpected condition!')
        
    # answer = pd.Series(likely_english_messages_raw).to_frame(english_message_field)
    if english_message_field in df:
        df = df.drop(english_message_field, axis=1)
        
    answer = df.join(pd.Series(likely_english_messages_raw).to_frame(english_message_field), on='id', how='left')

    return answer, inspect


#####################################
##### FUNCTIONS FOR SAVING DATA #####
#####################################


def save_data(df, database_filepath, table_name=DB_TABLE_NAME):
    """ Store DataFrame information into an SQLite3 database.

    Args:
        df (DataFrame): Source data to insert into database.
        database_filepath (str): Filename for the stored SQLite3 database.
        table_name (str, optional): Name of database table to write info into. Defaults to DB_TABLE_NAME.
    """
    print(f'Generating database {database_filepath}.')

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
        'message': f'VARCHAR({df.message.apply(lambda x: 0 if pd.isnull(x) else len(x)).max()})', # length of the longest 'message'
        'original': f'VARCHAR({df.original.apply(lambda x: 0 if pd.isnull(x) else len(x)).max()})', # length of the longest 'original'
        # 'message_raw': f'VARCHAR({df.message_raw.apply(lambda x: len(x)).max()})', # length of the longest 'message_raw'
        # 'message_english': f'VARCHAR({df.message_english.apply(lambda x: 0 if pd.isnull(x) else len(x)).max()})', # length of the longest 'message_english'
        'genre': f'VARCHAR({df.genre.apply(lambda x: len(x)).max()})', # length of the longest 'genre'
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