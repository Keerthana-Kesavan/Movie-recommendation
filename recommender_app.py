import streamlit as st
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle

# Title of the app
st.title('Movie Recommendation System')

# Upload the dataset
st.sidebar.header('Upload Dataset')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully!")
    st.write(df.head())

    # Configure Surprise Reader
    reader = Reader(rating_scale=(0.5, 5.0))
    
    # Create Surprise Dataset
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)

    # Build and train model
    model = SVD()
    model.fit(trainset)
    
    # Predict and evaluate model
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    st.write(f'Root Mean Squared Error: {rmse}')
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    st.write("Model saved successfully!")

    # Recommendation section
    st.sidebar.header('Get Recommendations')
    user_id = st.sidebar.text_input("User ID")
    movie_id = st.sidebar.text_input("Movie ID")

    if st.sidebar.button("Get Recommendation"):
        # Predict rating
        pred = model.predict(user_id, movie_id)
        st.write(f"Predicted rating for user {user_id} on movie {movie_id}: {pred.est:.2f}")

# Handling errors
except Exception as e:
    st.error(f"An error occurred: {e}")
