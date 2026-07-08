import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_data():
    print("Loading split datasets...")
    train_df = pd.read_csv('data/train_data.csv')
    test_df = pd.read_csv('data/test_data.csv')
    
    # Handle any potential missing values created during CSV save/load
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    return train_df, test_df

def train_and_evaluate(train_df, test_df):
    print("Vectorizing text data (TF-IDF)...")
    # Initialize the vectorizer (converts text to numbers)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    
    # Fit and transform the training data
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['label']
    
    # Transform the testing data
    X_test = vectorizer.transform(test_df['text'])
    y_test = test_df['label']

    print("Training Logistic Regression Model...")
    # Initialize and train the model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    print("Evaluating Model...")
    # Make predictions on the test set
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n--- Model Performance ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Fake (0)', 'True (1)']))
    
    return vectorizer, model

def save_artifacts(vectorizer, model):
    print("Saving model artifacts for deployment...")
    # Create a models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Save the vectorizer and the model
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    joblib.dump(model, 'models/logistic_model.pkl')
    print("Success! Artifacts saved in the 'models' folder.")

if __name__ == "__main__":
    try:
        train_data, test_data = load_data()
        tfidf_vec, trained_model = train_and_evaluate(train_data, test_data)
        save_artifacts(tfidf_vec, trained_model)
        print("\nPipeline Phase 2 Complete. Ready for Deployment.")
    except FileNotFoundError:
        print("Error: Could not find train_data.csv or test_data.csv. Did you run data_pipeline.py first?")