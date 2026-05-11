from utils import load_data, preprocess
from sklearn.metrics import accuracy_score

def train():
    df = load_data("../data/raw/sample_data.csv")
    df = preprocess(df)

    # Simple rule-based "model"
    df["prediction"] = ((df["visits"] > 7) | (df["age"] > 55)).astype(int)
    accuracy = accuracy_score(df["denied"], df["prediction"])
    
    print(f"Model accuracy: {accuracy:.2f}")
    print("Training complete. Sample output:")
    print(df.head())

if __name__ == "__main__":
    train()