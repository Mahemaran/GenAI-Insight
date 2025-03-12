import streamlit as st
import googleapiclient.discovery
import regex as re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
# from googletrans import Translator
from googletrans.client import Translator
import pandas as pd
import logging

# Function to fetch YouTube comments
def get_youtube_comments(video_id, api_key):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None
    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            pageToken=next_page_token,
            maxResults=100
        )
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return comments
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(message)s')
# Function to split text into batches
def batch_translate(texts, batch_size=1500):
    translator = Translator()
    translations = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # Create a batch
        try:
            batch_translations = [translator.translate(text, dest='en').text for text in batch]
            translations.extend(batch_translations)
        except Exception as e:
            logging.error(f"Error translating batch: {e}")
            translations.extend(["Error" for _ in batch])
    return translations
# Function to analyze sentiment using VADER
def get_sentiment(comment):
    sentiment_scores = analyzer.polarity_scores(comment)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return f'positive'
    elif compound_score <= -0.05:
        return f'negative'
    else:
        return f'neutral'

# Streamlit app
st.title("YouTube Comments Sentiment Analyzer")

api_key = "AIzaSyCbwvV9RgIX0B5vpe_0ki9SiHFwmi2a_9Y"
video_link = st.text_input("Enter YouTube Video link:")
# use REGex to extract ID
# def extract_video_id(url):
#     try:
#         if "v=" in url:
#             return url.split("v=")[1].split("&")[0]
#         else:
#             return None
#     except IndexError:
#         return None
def extract_video_id(url):
    # Define regex for YouTube video IDs
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|$)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return "No valid video ID found."
video_id = extract_video_id(video_link)
# Button to fetch and analyze comments
if st.button("Analyze Comments"):
    if not video_id:
        st.error("Please provide the Video ID.")
    else:
        with st.spinner("Fetching comments..."):
            try:
                comments = get_youtube_comments(video_id, api_key)
                if not comments:
                    st.warning("No comments found for the given video.")
                else:
                    df = pd.DataFrame(comments, columns=["Comment"])
                    df["translate"] = batch_translate(df["Comment"].tolist(), batch_size=500)
                    analyzer = SentimentIntensityAnalyzer()
                    df['Sentiment'] = df["translate"].apply(get_sentiment)
                    # Count of sentiments
                    sentiment_counts = df['Sentiment'].value_counts()
                    # Display data
                    st.success("Analysis complete!")
                    st.write(f"### Total Comments: {len(comments)}")
                    st.write("### Sentiment Distribution")
                    st.dataframe(df, height=400, width=2500)
                    # Plot sentiment distribution using Plotly
                    fig = px.bar(
                        sentiment_counts,
                        x=sentiment_counts.index,
                        y=sentiment_counts.values,
                        labels={'x': 'Sentiment', 'y': 'Count'},
                        title="Sentiment Count",
                        text_auto=True,
                        color=sentiment_counts.index,
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")