from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MaxAbsScaler

app = Flask(__name__, template_folder='template')

# Load the data and set up the recommendations model
df = pd.read_csv('anime.csv')

df.head()

df.isnull().sum()
episode_ = df['episodes'].replace('Unknown', np.nan)
episode_ = episode_.fillna(episode_.median())

type_ = pd.get_dummies(df['type'])
genre_ = df['genre'].str.get_dummies(sep=',')
genre_.head(10)
X = pd.concat([genre_, type_, episode_, df['rating'], df['members']], axis=1)
X.head(10)
scaled = MaxAbsScaler()
X = scaled.fit_transform(X)
recommendations = NearestNeighbors(n_neighbors=11).fit(X)
recommendations.kneighbors(X)
anime_indices = recommendations.kneighbors(X)[1]


@app.route("/", methods=["GET"])
def home():
    return render_template('index.html')


@app.route("/recommend", methods=['POST'])
def recommend():
    anime = request.form['anime']
    result = recommend_me(anime)
    return render_template('recommendations.html', result=result)


def get_index(x):
    return df[df['name'] == x].index.tolist()[0]


def recommend_me(anime):
    recommendations = []
    index = get_index(anime)
    for i in anime_indices[index][:]:
        recommendations.append(
            {'name': df.iloc[i]['name'],
             'genre': df.iloc[i]['genre'],
             'episodes': df.iloc[i]['episodes'],
             'rating': df.iloc[i]['rating']}
        )
    return recommendations


if __name__ == '__main__':
    app.debug = True
    app.run()
