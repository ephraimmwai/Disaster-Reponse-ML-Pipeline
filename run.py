import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objs as plt
from sklearn.externals import joblib
# import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''
    INPUT:
    text - message text

    OUTPUT:
    clean_tokens - tokenized text

    Tokenizes the messages
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens 

engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response_tweets', engine)

# load model
# model = joblib.load("models/classifier.pkl")
with open("models/classifier.pkl", 'rb') as f:
    # data = pickle.load(f)
    model = joblib.load(f)
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    '''
    Generated the summarised data visualization and render the home page
    '''    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # category count
    cats_df = df[df.columns.tolist()[4:]].sum().reset_index().rename(columns={'index':'cat_name', 0:'count'}).sort_values(by='count',ascending=False)
    cat_names,cat_counts = cats_df['cat_name'],cats_df['count']
    # create visuals
    graphs = [
        {
            'data': [
                plt.Pie(
                    labels=genre_names,
                    values=genre_counts,
                    marker_colors=['lightsteelblue','goldenrod','cadetblue']
                )
            ],

            'layout': {
                'title': 'Distribution of Message by Categories',
                'paper_bgcolor':'rgba(255, 1, 1, 0)'
            }
        },
        {
            'data': [
                plt.Bar(
                    x=cat_names,
                    y=cat_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message by Categories',
                'yaxis': {
                    'title': "Messages Count"
                },
                'xaxis': {
                    'title': "Category"
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
    '''
    Handle the message post data to classify the message using the classification model
    '''

    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001)


if __name__ == '__main__':
    main()