import pandas as pd

try:
    df = pd.read_csv('data/training_data.csv')
    print("CSV loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    print("\nSample questions and answers:")
    for i, row in df.head().iterrows():
        print(f"Q: {row['question']}")
        print(f"A: {row['answer']}")
        print()
except Exception as e:
    print(f"Error loading CSV: {str(e)}")