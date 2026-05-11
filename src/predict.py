from utils import load_data, preprocess

def predict():
    df = load_data("../data/raw/sample_data.csv")
    df = preprocess(df)

    df["prediction"] = (df["visits"] > 7).astype(int)

    print("Predictions:")
    print(df[["patient_id", "prediction"]])

if __name__ == "__main__":
    predict()