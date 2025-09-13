import streamlit as st
import joblib

# Load trained model + vectorizer
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🔮 Disaster Tweet Classifier")

tweet = st.text_area("Enter a tweet to check if it's disaster related:")

if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet!")
    else:
        # Transform text
        tweet_vec = vectorizer.transform([tweet])

        # Predict class
        pred = model.predict(tweet_vec)[0]

        # Get prediction probabilities
        proba = model.predict_proba(tweet_vec)[0]
        class_index = list(model.classes_).index(pred)  # map class label to index
        confidence = proba[class_index]

        # Map prediction to label text
        if pred == 1:
            label = "✅ Not Disaster"
        elif pred == 2:
            label = "🚨 Disaster"
        else:
            label = "❓ Unknown"

        # Show result
        st.success(f"Prediction: {label} (Confidence: {confidence:.2f})")
