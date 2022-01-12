import json
import re
from operator import itemgetter

import plotly
import pandas as pd
import joblib

from typing import List
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text: str) -> List[str]:
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('social_media_messages', engine)

# load model
model = joblib.load("models/disaster_response_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_names = df.columns[4:]
    category_counts = [df[column].sum() for column in category_names]
    categories_zipped = list(zip(category_names, category_counts))
    categories_sorted = sorted(categories_zipped, key=itemgetter(1), reverse=True)
    category_names_sorted, category_counts_sorted = map(list, zip(*categories_sorted))
    category_names_sorted = [category.replace('_', ' ').capitalize() for category in category_names_sorted]

    graph_3_data = {
        'direct': {},
        'news': {},
        'social': {},
    }
    for genre in genre_names:
        df_genre = df[df['genre'] == genre]
        graph_3_data[genre]['message_lengths'] = df_genre['message'].map(len).tolist()
        graph_3_data[genre]['number_categories'] = df_genre.iloc[:, 4:][df == 1].sum(axis=1).tolist()

    scatter_colors = {
        'direct': 'red',
        'news': 'green',
        'social': 'blue',
    }
    colors_direct = df[df['genre'] == 'direct']['genre'].map(scatter_colors)
    colors_news = df[df['genre'] == 'news']['genre'].map(scatter_colors)
    colors_social = df[df['genre'] == 'social']['genre'].map(scatter_colors)

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names_sorted,
                    y=category_counts_sorted
                )
            ],

            'layout': {
                'title': 'Count of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Scatter(
                    x=graph_3_data['direct']['message_lengths'],
                    y=graph_3_data['direct']['number_categories'],
                    mode='markers',
                    marker=dict(
                        color=colors_direct,
                    ),
                    name='Direct',
                ),
                Scatter(
                    x=graph_3_data['news']['message_lengths'],
                    y=graph_3_data['news']['number_categories'],
                    mode='markers',
                    marker=dict(
                        color=colors_news,
                    ),
                    name='News',
                ),
                Scatter(
                    x=graph_3_data['social']['message_lengths'],
                    y=graph_3_data['social']['number_categories'],
                    mode='markers',
                    marker=dict(
                        color=colors_social,
                    ),
                    name='Social',
                ),

            ],
            'layout': {
                'title': 'Message Length vs. Number of Attributed Categories',
                'yaxis': {
                    'title': 'Number of Attributed Categories'
                },
                'xaxis': {
                    'title': 'Message Length',
                    'type': 'log',
                },
            }
        }
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
