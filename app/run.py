"""
Ross Heaton github.com/rossheat

Runs an application which classifies disaster response messages.

Example usage:

>>> python run.py
"""

import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Load data from the SQL database 
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# Load model from pickle file 
model = joblib.load("../models/classifier.pkl")['model']


# Index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Data required for chart showing categories and count
    category_names = df.iloc[:,4:].columns
    category_boolean = (df.iloc[:,4:] != 0).sum().values
    
    # Data required for chart showing genre by aid related status
    aid_related = df[df['aid_related']==1].groupby('genre').count()['message']
    not_aid_related = df[df['aid_related']==0].groupby('genre').count()['message']
    genre_names = list(aid_related.index)
    
    
    # Create visuals
    graphs = [
          {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }
            }
        },               
         # Chart which shows genre by aid related status      
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=aid_related,
                    name = 'Aid-related'

                ),
                Bar(
                    x=genre_names,
                    y= not_aid_related,
                    name = 'Not aid-related'
                )
            ],

            'layout': {
                'title': 'Distribution of messages by genre by aid related status',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode' : 'group'
            }
        },
    ]
    
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '') 

    # Use model to predict classification for query
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