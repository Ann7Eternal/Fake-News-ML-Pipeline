import streamlit as st
import joblib
import time
import os
import csv
from datetime import datetime

# --- 1. Load Artifacts (Cached for Performance) ---
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        model = joblib.load('models/logistic_model.pkl')
        return vectorizer, model
    except FileNotFoundError:
        st.error("Model artifacts not found. Please run train.py first.")
        return None, None

tfidf_vec, trained_model = load_models()

# --- 2. Monitoring Logger Function ---
def log_prediction(input_length, prediction_label, confidence, latency):
    log_file = 'monitoring_log.csv'
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write headers if it's a new file
        if not file_exists:
            writer.writerow(['Timestamp', 'Input_Length_Chars', 'Prediction', 'Confidence_Score', 'Latency_Seconds'])
        
        # Log the live data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, input_length, prediction_label, f"{confidence:.4f}", f"{latency:.4f}"])

# --- 3. Streamlit Web Interface ---
st.set_page_config(page_title="Fake News Detector AI", page_icon="📰")

st.title("📰 Real-Time Fake News Detection Pipeline")
st.markdown("""
This application is the deployment node of an End-to-End ML Pipeline. 
It features **real-time inference latency tracking** and **live data logging** for model monitoring.
""")

st.divider()

# User Input
user_input = st.text_area("Paste a news article or headline here:", height=200)

if st.button("Analyze Article"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    elif tfidf_vec and trained_model:
        with st.spinner("Analyzing text..."):
            # Start Latency Timer
            start_time = time.time()
            
            # 1. Preprocess
            vectorized_input = tfidf_vec.transform([user_input])
            
            # 2. Predict
            prediction = trained_model.predict(vectorized_input)[0]
            probabilities = trained_model.predict_proba(vectorized_input)[0]
            
            # Stop Latency Timer
            end_time = time.time()
            latency = end_time - start_time
            
            # Process Results
            confidence = probabilities[prediction]
            result_label = "True News" if prediction == 1 else "Fake News"
            
            # 3. Log to Monitoring CSV
            log_prediction(len(user_input), result_label, confidence, latency)
            
            # 4. Display Results to User
            if prediction == 1:
                st.success(f"✅ **Prediction:** {result_label}")
            else:
                st.error(f"🚨 **Prediction:** {result_label}")
                
            st.info(f"**System Metrics:** \n* Confidence Score: {confidence * 100:.2f}% \n* Inference Latency: {latency * 1000:.2f} ms")
            
st.divider()
st.caption("MLOps & System Design Showcase")