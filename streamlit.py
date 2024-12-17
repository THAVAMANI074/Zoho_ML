# Import necessary libraries
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Load the trained model
model_file = "audience_rating_model-final.pkl"
model = joblib.load(model_file)

# Title and description for Streamlit app
st.title("🎬 Audience Rating Prediction App")
st.write("Use this tool to predict the audience rating for a movie based on adjustable tomatometer ratings and other movie details.")

# Function to create movie data
def create_movie_data(tomatometer_rating):
    """
    Prepare a DataFrame with user-input movie data and a given tomatometer rating.
    """
    data = {
        "movie_title": [st.session_state.movie_title],
        "movie_info": [st.session_state.movie_info],
        "critics_consensus": [st.session_state.critics_consensus],
        "rating": [st.session_state.rating],
        "genre": [st.session_state.genre],
        "directors": [st.session_state.directors],
        "writers": [st.session_state.writers],
        "cast": [st.session_state.cast],
        "in_theaters_date": [st.session_state.in_theaters_date],
        "on_streaming_date": [st.session_state.on_streaming_date],
        "runtime_in_minutes": [st.session_state.runtime],
        "studio_name": [st.session_state.studio_name],
        "tomatometer_status": [st.session_state.tomatometer_status],
        "tomatometer_rating": [tomatometer_rating],
        "tomatometer_count": [st.session_state.tomatometer_count],
    }
    return pd.DataFrame(data)

# User inputs in the sidebar
st.sidebar.header("📋 Enter Movie Details")
st.sidebar.text_input("Movie Title", "Blockbuster Movie", key="movie_title")
st.sidebar.text_area("Movie Info", "An outstanding critically acclaimed movie", key="movie_info")
st.sidebar.text_input("Critics Consensus", "Overwhelmingly positive reviews", key="critics_consensus")
st.sidebar.selectbox("Rating", ["G", "PG", "PG-13", "R", "NC-17"], index=2, key="rating")
st.sidebar.text_input("Genre", "Drama", key="genre")
st.sidebar.text_input("Director(s)", "Famous Director", key="directors")
st.sidebar.text_input("Writer(s)", "Top Screenwriter", key="writers")
st.sidebar.text_input("Cast", "Famous Actor A, Famous Actor B", key="cast")
st.sidebar.date_input("In Theaters Date", key="in_theaters_date")
st.sidebar.date_input("On Streaming Date", key="on_streaming_date")
st.sidebar.number_input("Runtime (in minutes)", min_value=1, max_value=500, value=150, key="runtime")
st.sidebar.text_input("Studio Name", "Top Studio", key="studio_name")
st.sidebar.selectbox("Tomatometer Status", ["Fresh", "Certified Fresh", "Rotten"], index=1, key="tomatometer_status")
st.sidebar.number_input("Tomatometer Count", min_value=1, value=550, key="tomatometer_count")

# Slider for tomatometer ratings
st.subheader("Adjust Tomatometer Ratings")
tomatometer_range = st.slider("Select a range of Tomatometer Ratings", 85, 100, (85, 100))
target_audience_rating = st.number_input("Target Audience Rating", min_value=1, max_value=100, value=95)

# Generate predictions
if st.button("Predict Optimal Tomatometer Rating"):
    # Generate a range of tomatometer ratings
    tomatometer_ratings = np.arange(tomatometer_range[0], tomatometer_range[1] + 1, 1)
    repeated_data = pd.concat([create_movie_data(r) for r in tomatometer_ratings], ignore_index=True)
    
    # Predict audience ratings
    predicted_ratings = model.predict(repeated_data)
    
    # Find the best tomatometer rating
    differences = np.abs(predicted_ratings - target_audience_rating)
    best_index = np.argmin(differences)
    
    # Extract the best features and predictions
    best_input_features = repeated_data.iloc[best_index]
    best_prediction = predicted_ratings[best_index]
    
    # Display results
    st.subheader("🎯 Best Prediction Results")
    st.write(f"**Optimal Tomatometer Rating:** {best_input_features['tomatometer_rating']}")
    st.write(f"**Predicted Audience Rating:** {best_prediction:.2f}")
    
    st.write("**Final Adjusted Movie Features:**")
    st.dataframe(best_input_features.to_frame().T)
