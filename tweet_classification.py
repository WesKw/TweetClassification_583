import pandas as pd
import scipy

from time import time
from argparse import ArgumentParser
from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_precision():
    ...


def calculate_recall():
    ...


def calculate_fscore():
    ...


def calculate_accuracy():
    ...


def load_tweet_data(input: str) -> dict:
    """
    Loads tweets into a pandas dataframe. Input is assumed to be an xlsx file.
    """
    obama_df = pd.DataFrame()
    romney_df = pd.DataFrame()
    if ".xlsx" in input:
        data = pd.read_excel(input, sheet_name=None, header=0)
        obama_df = data["Obama"]
        romney_df = data["Romney"]

    return {"Obama": {"df": obama_df, "classifier": None}, "Romney": {"df": romney_df, "classifier": None}}


def build_classifier(df: pd.DataFrame):
    start = time()
    text_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english")

    tweets = df["Anotated Tweet"]
    classes = df["Class"]

    return None


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data.drop(index=0)
    df = df.set_axis(labels=['None', 'date', 'time', 'Anotated Tweet', 'Class', 'Yourclass'], axis=1) # fix labels
    df = df.drop(labels=['None', 'Yourclass'], axis=1) # drop irrelevant labels
    df = df[(df["Class"] != 2) & (df["Class"] != "!!!!") & (df["Class"] != "irrevelant")] # remove any classes that are not 0, 1, -1

    # todo:: fix datetimes?

    # return cleaned df
    return df


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("training", help="The training data to use for tweet classification.")
    args.add_argument("test", help="The test data used to determine accuracy.")
    args.add_argument("-o", "--output", help="The file to save the output to.")
    opts = args.parse_args()

    training_dfs = load_tweet_data(opts.training)

    for key,df_data in training_dfs.items():
        df_data["df"] = clean_data(df_data["df"])

        # build the classifier
        classifier = build_classifier(df_data["df"])


    # load the test data after we train our model
    # test_dfs = load_tweet_data(opts.test)