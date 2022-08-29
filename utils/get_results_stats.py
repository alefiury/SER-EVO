import pandas as pd

def main():
    df = pd.read_csv("../scores/Test-Pop500-batch_1024-gen_50.csv")

    df = df[["Acc", "Precision", "Recall", "F1_Score"]]

    print(df.describe())


if __name__ == "__main__":
    main()