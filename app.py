from flask import Flask, render_template, request
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from googleapiclient.discovery import build

import googleapiclient.discovery

app = Flask(__name__)

# Function to fetch comments from YouTube API
def google_api(video_id):
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyBKspZBSYbb_mnTOY-UyOBl4R2MLz5wgls"

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="id,snippet",
        maxResults=300,
        order="relevance",
        videoId=video_id
    )

    response = request.execute()
    return response

# Function to perform sentiment analysis and recommend top comments
def analyze_and_recommend(video_id, threshold=0.05):
    # Fetch comments from YouTube API
    response = google_api(video_id)

    # Create DataFrame from comments
    authorname = []
    comments = []
    for i in range(len(response["items"])):
        authorname.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"])
        comments.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["textOriginal"])

    df_1 = pd.DataFrame(comments, index=authorname, columns=["Comments"])

    # Remove NaN values and empty strings
    df_1.dropna(inplace=True)
    blanks = [i for i, comment in df_1.itertuples() if type(comment) == str and comment.isspace()]
    df_1.drop(blanks, inplace=True)

    # Sentiment analysis using VADER with flexible threshold
    sid = SentimentIntensityAnalyzer()
    df_1['Scores'] = df_1['Comments'].apply(lambda x: sid.polarity_scores(x))
    df_1['Compound'] = df_1['Scores'].apply(lambda score_dict: score_dict['compound'])
    df_1['Polarity'] = df_1['Compound'].apply(lambda c: 'neutral' if -threshold < c < threshold else ('negative' if c < -threshold else 'positive'))

    # Calculate sentiment counts
    sentiment_counts = df_1['Polarity'].value_counts()

    # Recommend top comments for each sentiment
    def recommend_top_comments(sentiment, top_n=5):
        top_comments = df_1[df_1['Polarity'] == sentiment].nlargest(top_n, 'Compound')
        return top_comments[['Comments', 'Compound']]

    # Recommend and display top positive comments
    top_positive_comments = recommend_top_comments('positive')

    # Recommend and display top negative comments
    top_negative_comments = recommend_top_comments('negative')

    # Recommend and display top neutral comments
    top_neutral_comments = recommend_top_comments('neutral')

    # Recommendation dictionary to pass to the template
    recommendations = {
        'positive': top_positive_comments.to_dict(orient='records'),
        'negative': top_negative_comments.to_dict(orient='records'),
        'neutral': top_neutral_comments.to_dict(orient='records')
    }

    return sentiment_counts.to_dict(), recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_id = request.form['video_id']
        sentiment_counts, recommendations = analyze_and_recommend(video_id)
        return render_template('index.html', video_id=video_id, sentiment_counts=sentiment_counts, recommendations=recommendations)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
