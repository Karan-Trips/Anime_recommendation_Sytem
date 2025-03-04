import matplotlib
matplotlib.use('Agg')  # Ensure non-GUI backend is used

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from flask import Flask, render_template, request
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MaxAbsScaler

app = Flask(__name__, template_folder='template')

# Ensure the static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Load dataset
df = pd.read_csv('/Users/hyperlink/Desktop/Fluttertonative/Anime_recommendation_Sytem/Anime Recommendation/anime.csv')

# Print columns to debug missing 'type'
print("columns==>> {df.columns}")
print(df.info())
print(df.value_counts())

# Handle missing columns, specifically 'type'
if 'type' in df.columns:
    type_ = pd.get_dummies(df['type'])
else:
    print("Column 'type' not found, creating an empty DataFrame for 'type'")
    type_ = pd.DataFrame()

# Handle missing and non-numeric values in 'episodes'
episode_ = df['episodes'].replace('Unknown', np.nan)
episode_ = pd.to_numeric(episode_, errors='coerce')
episode_ = episode_.fillna(episode_.median())

# Prepare data
genre_ = df['genre'].str.get_dummies(sep=',')
X = pd.concat([genre_, type_, episode_, df['rating'], df['members']], axis=1)

# Normalize data
scaled = MaxAbsScaler()
X = scaled.fit_transform(X)

# Load or fit the NearestNeighbors model
try:
    recommendations = joblib.load('recommendations_model.pkl')
except FileNotFoundError:
    recommendations = NearestNeighbors(n_neighbors=10).fit(X)
    joblib.dump(recommendations, 'recommendations_model.pkl')

# Get anime indices
anime_indices = recommendations.kneighbors(X)[1]

@app.route("/", methods=["GET"])
def home():
    return render_template('index.html')

@app.route("/recommend", methods=['POST'])
def recommend():
    anime = request.form['anime'].lower()
    result = recommend_me(anime)
    return render_template('recommendations.html', result=result)

def get_index(x):
    return df[df['name'].str.lower() == x.lower()].index.tolist()[0]

def recommend_me(anime):
    recommendations = []
    anime = anime.lower()  
    index = get_index(anime)
    distances, indices = recommendations.kneighbors([X[index]])  # Neighbors and distances
    for i in indices[0]:  # Iterate through neighbors' indices
        recommendations.append(
            {'name': df.iloc[i]['name'],
             'genre': df.iloc[i]['genre'],
             'episodes': df.iloc[i]['episodes'],
             'rating': df.iloc[i]['rating']}
        )
    return recommendations

@app.route('/plots')
def ploting():
    # Popular anime bar chart
    plt.figure(figsize=(15, 7))
    plt.ylim(0.0, 10.0)
    plt.bar(df['name'].head(5), df['rating'].head(5))
    plt.xlabel("Popularity")
    plt.title("Popular anime")
    plt.savefig('static/barchart.jpg')  # Save figure without displaying it

    # Top genres with the highest total members
    genre_member = df.groupby("genre", as_index=False)["members"].sum()
    genre_member.sort_values(by="members", ascending=False, inplace=True)
    plt.figure(figsize=(10, 6))
    top_3_genres = genre_member.head(3)
    plt.bar(top_3_genres["genre"], top_3_genres["members"])
    plt.xlabel("Genre")
    plt.ylabel("Total Members")
    plt.title("Top 5 Genres with Highest Total Members")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/barchart2.jpg')  # Save figure without displaying it

    # Rating vs Members scatter plot
    plt.figure()
    g = sns.scatterplot(data=df, x="members", y="rating", hue="type")
    g.set_title("Rating vs Members by Type")
    g.set_xlabel("Members")
    g.set_ylabel("Rating")
    plt.savefig('static/scatterplot.jpg')  # Save figure without displaying it

    # Pie chart for Anime Types
    types = pd.DataFrame(df["type"].value_counts())
    types.plot.pie(y="type", figsize=(8, 8), autopct="%1.1f%%")
    plt.legend()
    plt.title("Distribution of Anime Type")
    plt.savefig('static/piechart.jpg')  # Save figure without displaying it

    # Count plot for Anime Types
    plt.figure(figsize=(10,10))
    sns.countplot(x='type', data=df)
    plt.savefig('static/countplot.jpg')  # Save figure without displaying it

    return render_template('plots.html')

if __name__ == '__main__':
    app.run(debug=False)  # Ensure Flask runs without debug mode to avoid threading issues
