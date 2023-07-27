from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MaxAbsScaler
import seaborn as sns
import matplotlib.pyplot as plt

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
recommendations = NearestNeighbors(n_neighbors=10).fit(X)
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

@app.route('/plots')
def ploting():
  
    plt.figure(figsize=(15, 7))
    plt.ylim(0.0, 10.0)
    plt.bar(df['name'].head(5), df['rating'].head(5))
    plt.xlabel("Popularity")
    plt.title("Popular anime")
    plt.savefig('static/barchart.jpg')

    genre_member = df.groupby("genre", as_index=False)["members"].sum()
    genre_member.sort_values(by="members", ascending=False, inplace=True)
    genre_member.head(3)

    plt.figure()
    g = sns.scatterplot(data=df, x="members", y="rating", hue="type")
    g.set_title("Rating vs Members by Type")
    g.set_xlabel("Members")
    g.set_ylabel("Rating")
    plt.savefig('static/scatterplot.jpg')

    types = pd.DataFrame(df["type"].value_counts())
    types.plot.pie(y="type", figsize=(8, 8), autopct="%1.1f%%")
    plt.legend()
    plt.title("Distribution of Anime Type")
    plt.savefig('static/piechart.jpg')

    return render_template('plots.html')


if __name__ == '__main__':
    app.debug = True
    app.run()
