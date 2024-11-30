import sys
import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


# Downloads for NLTK tools
nltk.download('words', quiet=True);
nltk.download('wordnet', quiet=True);
nltk.download('punkt_tab', quiet=True);
nltk.download('stopwords', quiet=True);


def display_results(Y_test, Y_pred, average='micro'):
    """ Display R1, Accuracy and Precision scores for predictions vs observed values.

    Args:
        Y_test (DataFrame): Observed Y data.
        Y_pred (DataFrame or array): Predicted Y data.
        average (str, optional): Method to use in F1 score. Defaults to 'micro'.
    """
    f1 = f1_score(Y_test, Y_pred, average=average, zero_division=0)
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average=average, zero_division=0) 

    print(f"F1: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")


def load_data(database_filepath, db_table_name='project2'):
    """ Load source data from SQLite3 database, and separate into X and Y data for use in classification model.

    Args:
        database_filepath (str): Filepath to SQLite3 database file.
        db_table_name (str, optional): Table to query in the database. Defaults to 'project2'.

    Returns:
        tuple: Three-tuple of X data (DataFrame), Y data (DataFrame), and category names (list of str).
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


def tokenize(
        text, 
        lemmatize=True, 
        make_lowercase=True, 
        remove_non_words=False, 
        remove_stop_words=True, 
        verbose=False,
        ):
    """ Custom function to tokenize strings contained in text, and convert the resulting 
    tokens back into strings so they can be further processed by TfidfVectorizer class.

    Args:
        text (Series): Series of strings to parse.
        lemmatize (bool, optional): Apply lemmatization to tokens. Defaults to True.
        make_lowercase (bool, optional): Make all tokens lower case. Defaults to True.
        remove_non_words (bool, optional): Remove non-English words from tokens. Defaults to True.
        remove_stop_words (bool, optional): Remove stop words from tokens. Defaults to True.
        verbose (bool, optional): Print messages to screem. Defaults to False.
    """
    def get_tokens_per_row(text, make_lowercase):
        if verbose:
            print('... tokenizing and converting to lowercase.' if make_lowercase else '... tokenizing.')
        return [nltk.word_tokenize(s.lower() if make_lowercase else s) for s in text]
    
    def _remove_non_words(tokens_by_message):
        if verbose:
            print('... removing non-words.')
        answer = []
        for tokens in tokens_by_message:
            answer.append([i for i in tokens if i.isalpha()])
        return answer

    def _remove_stop_words(tokens_by_message, stop_words=set(nltk.corpus.stopwords.words('english'))):
        if verbose:
            print('... removing stop words.')
        answer = []
        for tokens in tokens_by_message:
            answer.append([i for  i in tokens if i not in stop_words])
        return answer
    
    def _lemmatize(tokens_by_message, lemmatizer=nltk.stem.WordNetLemmatizer()):
        if verbose:
            print('... lemmatizing.')
        answer = []
        for tokens in tokens_by_message:
            answer.append([lemmatizer.lemmatize(i).strip() for i in tokens])
        return answer

    tokens_by_message = get_tokens_per_row(text, make_lowercase)

    if lemmatize:
        tokens_by_message = _lemmatize(tokens_by_message)

    if remove_stop_words:
        tokens_by_message = _remove_stop_words(tokens_by_message)
        
    if remove_non_words:
        tokens_by_message = _remove_non_words(tokens_by_message)

    # now combine back to strings (one per row) so we can pass it back to TfidfTransformer
    processed_messages = [' '.join(tokens) for tokens in tokens_by_message]

    return processed_messages


def build_model(
        use_model=RandomForestClassifier(random_state=42, n_jobs=-1), 
        grid_search_params={}, 
        ):
    """ Build a classification model using a Scitkit-Learn pipline to procss text and 
    generate a TF-IDF matrix. Optionally apply a paramter grid search.

    Thank you: https://stackoverflow.com/questions/67768470/valueerror-found-input-variables-with-inconsistent-numbers-of-samples-6-80

    Args:
        use_model (object, optional): Classifier model to utilize. Defaults to RandomForestClassifier(random_state=42, n_jobs=-1).
        grid_search_params (dict, optional): Dictionary of paramters and associated grid search values. Defaults to {}.

    Returns:
        object: GridSearchCV object containing the classification model.
    """

    transformer = ColumnTransformer([
        ('vect', TfidfVectorizer(tokenizer=tokenize, token_pattern=None), 'message'),
        ('genre_onehot', OneHotEncoder(dtype='int'), ['genre']),
    ], remainder='drop')

    pipeline = Pipeline([
        ('vect', transformer),    
        ('model', use_model),
    ])
        
    return GridSearchCV(pipeline, grid_search_params, n_jobs=-1)



def evaluate_model(model, X_test, Y_test, category_names):
    """ Generates a classification report for the fit model.

    Args:
        model (object): A fit classication model.
        X_test (_type_): Test X data to use for preiction.
        Y_test (_type_): Test Y data (observed) to compare against prediction.
        category_names (list of str): Target (Y) category labels.
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names, zero_division=0.0))


def save_model(model, model_filepath):
    """ Store fit model as a pickle object.

    Args:
        model (object): A fit classification model.
        model_filepath (str): Filepath for pickle file.
    """
    print(f'Pickling model as {model_filepath}.')
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    print('... finished!')


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