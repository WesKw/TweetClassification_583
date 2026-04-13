import pandas as pd

from argparse import ArgumentParser


CLASS_MAPPING = {
    "Negative": -1,
    "Positive": 1,
    "Mixed": 0,
    "Neutral": 2
}


def determine_accuracy(test, result, name=""):
    total_count = len(test["Class"])
    test["Your Class"] = result["Class"]
    test["Correct"] = (test["Class"] == test["Your Class"])
    test["Correct"] = test["Correct"].astype("int")
    correct = test["Correct"].sum()
    print(f"Accuracy for {name}: {correct/total_count:.02f}")


def clean_data(data: pd.DataFrame, ignore_class=False) -> pd.DataFrame:
    df = data.drop(index=0)
    df = df.set_axis(labels=['None', 'date', 'time', 'Anotated Tweet', 'Class', 'Yourclass'], axis=1) # fix labels
    df = df.drop(labels=['None', 'Yourclass'], axis=1) # drop irrelevant labels
    df["Class"] = df["Class"].astype("string")
    # if not ignore_class:
    #     df = df[df["Class"].isin(VALID_CLASSES)] # remove any rows that have bad class labels
    df = df.dropna()

    # todo:: fix datetimes?

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
    args.add_argument("test_input", help="The test data containing classes.")
    args.add_argument("results", help="The test data used to determine accuracy.")
    args.add_argument("--test-has-labels", action="store_true", help="If the test data has class labels, use these to calculate evaluation metrics.")
    args.add_argument("-o", "--output", help="The file to save the output to.")
    opts = args.parse_args()

    test_df = load_tweet_data(opts.test_input)
    obama_df = clean_data(test_df["Obama"]).reset_index()
    romney_df = clean_data(test_df["Romney"]).reset_index()
    print(obama_df)
    print(romney_df)
    results_df = load_tweet_data(opts.results)
    obama_results = results_df["Obama"].set_axis(labels=["Tweet", "Class"], axis=1)
    obama_results["Class"] = obama_results["Class"].map(lambda x: CLASS_MAPPING[x])
    obama_results["Class"] = obama_results["Class"].astype("string")
    romney_results = results_df["Romney"].set_axis(labels=["Tweet", "Class"], axis=1)
    romney_results["Class"] = romney_results["Class"].map(lambda x: CLASS_MAPPING[x])
    romney_results["Class"] = romney_results["Class"].astype("string")
    print(obama_results)
    print(romney_results)

    determine_accuracy(obama_df, obama_results, "Obama")
    determine_accuracy(romney_df, romney_results, "Romney")