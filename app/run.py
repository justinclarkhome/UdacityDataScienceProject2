import json
import plotly
import plotly.express as px
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
import joblib
from sqlalchemy import create_engine
import plotly.graph_objs as go


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('project2', engine).set_index('id')

# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # count of True values across categories for each message
    categories = [k for k, v in df.dtypes.items() if v in [int, np.int64]]
    cat_counts_per_message = df[categories].sum(axis=1).reset_index(drop=True)
    histogram_data, histogram_labels = np.histogram(cat_counts_per_message)
    hist_plot_data = pd.concat({
        'labels': pd.Series(histogram_labels).apply(lambda x: f'{x:.0f}'),
        'hist': pd.Series(histogram_data),
    }, axis=1).dropna().set_index('labels')

    counts_per_category = df[categories].sum().sort_values(ascending=False)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=list(counts_per_category.index),
                    y=counts_per_category,
                )
            ],
            'layout': {
                'title': 'Histogram of True Values per Category',
                'xaxis': {'tickangle': 45},
                'yaxis': {'title': 'Frequency of True Values in Category'},
            }
        },
        {
            'data': [
                Bar(
                    x=list(hist_plot_data.index),
                    y=hist_plot_data.squeeze(),
                )
            ],
            'layout': {
                'title': 'Histogram of True Category Values per Message',
                'xaxis': {'title': 'Number of True Categories per Message'},
                'yaxis': {'title': 'Frequency'},
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                )
            ],
            'layout': {
                'title': 'Number of Messages by Genre',
                'xaxis': {'title': 'Genre'},
                'yaxis': {'title': 'Frequency'},
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()