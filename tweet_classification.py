import pandas as pd

from argparse import ArgumentParser


def load_tweets(input: str) -> list:
    """
    Loads tweets into a pandas dataframe. Input is assumed to be an xlsx file.
    """
    obama_df = pd.DataFrame()
    romney_df = pd.DataFrame()
    if ".xlsx" in input:
        data = pd.read_excel(input, sheet_name=None)
        # print(data)
        obama_df = data["Obama"]
        romney_df = data["Romney"]

    return [obama_df, romney_df]


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data[(data["Unnamed: 4"] != 2)]
    # data = data.drop(labels="Unnamed: 5")
    
    return data


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