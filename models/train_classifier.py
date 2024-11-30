import sys
import os
import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

# Downloads for NLTK tools
nltk.download('words', quiet=True);
nltk.download('wordnet', quiet=True);
nltk.download('punkt_tab', quiet=True);
nltk.download('stopwords', quiet=True)

def display_results(Y_test, Y_pred, average='macro'):
    f1 = f1_score(Y_test, Y_pred, average=average, zero_division=0)
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average=average, zero_division=0) 

    print(f"F1: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")


# def get_english_words_in_string(s, english_word_set=set(nltk.corpus.words.words()), adhoc_words=()):
#     """_summary_

#     Args:
#         s (str): String to parse.
#         english_words (set, optional): A set of known English words. Defaults to set(nltk.corpus.words.words()).
#         adhoc_words (collection, optional): Additional words to consider as English, along with english_word_set.

#     Returns:
#         tuple: Three-tuple containing ths set of all words in the string, the set of all English words in the string, and the ratio.
#     """

#     all_words = [i for i in nltk.wordpunct_tokenize(s.lower()) if i.isalpha()]
#     english_words = [i for i in all_words if i in english_word_set or i in adhoc_words]
#     if all_words:
#         ratio_of_english_words = len(set(english_words))/len(set(all_words))
#     else:
#         ratio_of_english_words = None
#     return all_words, english_words, ratio_of_english_words



def load_data(database_filepath, db_table_name='project2'):
    """_summary_

    Args:
        database_filepath (_type_): _description_
        db_table_name (str, optional): _description_. Defaults to 'project2'.

    Returns:
        _type_: _description_
    """
    print(database_filepath)
    conn = sqlite3.connect(database_filepath)
    cur = conn.cursor()
    df = pd.read_sql(f'SELECT * from {db_table_name}', con=conn, index_col='id')
    conn.close()

    category_names = [k for k,v in df.dtypes.items() if v in [int, np.int64]]
    assert len(category_names) == 36, f'There should be 36 categories (got {len(category_names)})'

    Y = df[category_names]
    X = df.drop(category_names, axis=1)

    return X, Y, category_names


def tokenize(text):
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    pass


def main_example():
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