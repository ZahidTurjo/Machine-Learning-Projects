import streamlit as st
import pickle
import pandas as pd


with open('music.pkl', 'rb') as file:
    df, cosine_sim = pickle.load(file)

def recommend_songs(song_name, cosine_sim=cosine_sim, df=df, top_n=5):

    idx = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        return "Song not found in the dataset!"
    idx = idx[0]

 
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]


    song_indices = [i[0] for i in sim_scores]

    # top 5 similar songs
    return df[['artist', 'song']].iloc[song_indices]


st.title("🎶 Song Recommendation System")
st.write("Enter a song title to get recommendations:")

song_list = sorted(df['song'].dropna().unique())
song_input = st.selectbox("🎵 Select a song:", song_list)

if st.button("Get Recommendations"):
    if song_input:
        recommendations = recommend_songs(song_input)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.subheader(f"Recommended songs similar to '{song_input}':")
            st.table(recommendations)
    else:
        st.warning("Please enter a song title.")
