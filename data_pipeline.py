import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_prepare_data():
    print("Loading raw data...")
    # Load the CSV files
    fake_df = pd.read_csv('data/Fake.csv')
    true_df = pd.read_csv('data/True.csv')

    # Add labels (0 for Fake, 1 for True)
    fake_df['label'] = 0
    true_df['label'] = 1

    # Combine the datasets
    print("Combining and shuffling datasets...")
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # We only need the 'text' and 'label' columns for this MVP
    df = df[['text', 'label']]
    
    # Drop any empty rows
    df = df.dropna()

    # Shuffle the dataset to ensure a good mix of fake and real news
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # To ensure training happens fast (within your 24hr window), 
    # we will take a random subsample of 10,000 articles. 
    df_sample = df.head(10000)
    
    return df_sample

def split_and_save_data(df):
    print("Splitting data into train and test sets...")
    # Split: 80% for training, 20% for testing
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save to the data folder
    train_df.to_csv('data/train_data.csv', index=False)
    test_df.to_csv('data/test_data.csv', index=False)
    
    print(f"Success! Saved train_data.csv ({len(train_df)} rows) and test_data.csv ({len(test_df)} rows).")

if __name__ == "__main__":
    # Ensure the data folder exists
    if not os.path.exists('data'):
        os.makedirs('data')
        
    try:
        dataset = load_and_prepare_data()
        split_and_save_data(dataset)
        print("Data pipeline executed successfully. Ready for Phase 2: Training.")
    except FileNotFoundError:
        print("Error: Could not find Fake.csv or True.csv. Please ensure they are inside the 'data' folder.")