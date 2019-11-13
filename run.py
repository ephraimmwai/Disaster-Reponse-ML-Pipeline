import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
# import joblib
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

# load data
<<<<<<< HEAD:run.py
# engine = create_engine('sqlite:///../data/DisasterResponse.db')
# df = pd.read_sql_table('disaster_response_tweets', engine)
# engine = create_engine('sqlite:///../data/DisasterResponse.db')
# path = '/home/ubuntu/disaster-response-env/disaster-reponse-environment/'
# path = 'C:/Users/lenovo/OneDrive - Centum Investment Company Limited/Food/disaster_response_pipeline_project/'
=======
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response_tweets', engine)

# load model
model = joblib.load("../models/classifier.pkl")
>>>>>>> dfd1c71c382a4c997f75e7cba24fbe6f55c7f262:app/run.py

engine = create_engine(r'sqlite:///data/DisasterResponse.db')
df = pd.read_sql_query('select * from "disaster_response_tweets"', con=engine)

# load model
# model = joblib.load("models/classifier.pkl")
with open("models/classifier.pkl", 'rb') as f:
    # data = pickle.load(f)
    model = joblib.load(f)
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message by Genres',
                'yaxis': {
                    'title': "Messages Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
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
    app.run(host='0.0.0.0', port=3001)


if __name__ == '__main__':
    main()