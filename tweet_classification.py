import pandas as pd
import scipy

from argparse import ArgumentParser


def load_tweets(input: str) -> list:
    """
    Loads tweets into a pandas dataframe. Input is assumed to be an xlsx file.
    """
    obama_df = pd.DataFrame()
    romney_df = pd.DataFrame()
    if ".xlsx" in input:
        data = pd.read_excel(input, sheet_name=None, header=0)
        # print(data)
        obama_df = data["Obama"]
        romney_df = data["Romney"]

    return [obama_df, romney_df]


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data.drop(index=0)
    df = df.set_axis(labels=['None', 'date', 'time', 'Anotated Tweet', 'Class', 'Yourclass'], axis=1)
    # print(df.columns)
    df = df.drop(labels=['None', 'Yourclass'], axis=1)
    df = df[(df["Class"] != 2) & (df["Class"] != "!!!!") & (df["Class"] != "irrevelant")] # remove any classes that are not 0 1 or -1
    # data = data[(data["Unnamed: 4"] != 2)]
    # data = data.drop(labels="Unnamed: 5")
    
    return df


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("training", help="The training data to use for tweet classification.")
    args.add_argument("test", help="The test data used to determine accuracy.")
    args.add_argument("-o", "--output", help="The file to save the output to.")
    opts = args.parse_args()

    data = load_tweets(opts.training)

    for df in data:
        df = clean_data(df)
        print(df)