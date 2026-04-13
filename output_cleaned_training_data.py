import pandas as pd

from argparse import ArgumentParser


VALID_CLASSES=['-1', '0', '1']


def clean_data(data: pd.DataFrame, ignore_class=False) -> pd.DataFrame:
    df = data.drop(index=0)
    df = df.set_axis(labels=['None', 'date', 'time', 'Anotated Tweet', 'Class', 'Yourclass'], axis=1) # fix labels
    df = df.drop(labels=['None', 'Yourclass'], axis=1) # drop irrelevant labels
    df["Class"] = df["Class"].astype("string")
    if not ignore_class:
        df = df[df["Class"].isin(VALID_CLASSES)] # remove any rows that have bad class labels
    df = df.dropna()

    # remove class column for testing
    df = df.drop(labels="Class", axis=1)

    # return cleaned df
    return df


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

    return {"Obama": obama_df, "Romney": romney_df}


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("training", help="The training data to use for tweet classification.")
    opts = args.parse_args()
    data = load_tweet_data(opts.training)
    obama_df = clean_data(data["Obama"])
    romney_df = clean_data(data["Romney"])

    writer = pd.ExcelWriter('training_data_for_llm.xlsx', engine='xlsxwriter')
    # writer.sheets["Obama"] = writer.book.add_worksheet("Obama")
    # writer.sheets["Romney"] = writer.book.add_worksheet("Romney")
    obama_df.to_excel(writer, sheet_name="Obama")
    romney_df.to_excel(writer, sheet_name="Romney")
    writer.close()
    