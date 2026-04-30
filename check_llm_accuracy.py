import pandas as pd

from argparse import ArgumentParser


CLASS_MAPPING = {
    "Negative": -1,
    "Positive": 1,
    "Neutral": 0,
    "Mixed": 2,
    "positive": 1,
    "negative": -1,
    "neutral": 0,
    "mixed": 2
}


def determine_accuracy(test, result, name=""):
    total_count = len(test["Class"])
    test["Predicted Class"] = result["Class"]
    test["Correct"] = (test["Class"] == test["Predicted Class"])
    test["Correct"] = test["Correct"].astype("int")
    correct = test["Correct"].sum()
    print(f"Accuracy for {name}: {correct/total_count:.02f}")


def load_input_data(input: str) -> dict:
    """
    Loads tweets into a pandas dataframe. Input is assumed to be an xlsx file.
    """
    df = pd.DataFrame()
    if ".xlsx" in input:
        data = pd.read_excel(input, sheet_name=None, header=None)
        df = data["Sheet1"]
        df = df.set_axis(labels=[0, "Tweet", 2, 3, 4, "5"], axis=1)
        df["Class"] = df["5"]
        df.drop(columns=[0, 2, 3, 4, "5"], inplace=True)
        print(df)

    return df


def load_result_data(input: str) -> dict:
    """
    Loads results into a pandas dataframe. Input is assumed to be an xlsx file.
    """
    df = pd.DataFrame()
    if ".xlsx" in input:
        data = pd.read_excel(input, sheet_name=None, header=0)
        results_df = data["Tweet Classifications"]
        results_df["Class"] = results_df["Class"].map(lambda x: CLASS_MAPPING[x])
        results_df["Class"] = results_df["Class"].astype("int")
        print(results_df)

    return results_df


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("test_input", help="The test data containing classes.")
    args.add_argument("results", help="The test data used to determine accuracy.")
    args.add_argument("--test-has-labels", action="store_true", help="If the test data has class labels, use these to calculate evaluation metrics.")
    args.add_argument("-o", "--output", help="The file to save the output to.")
    opts = args.parse_args()

    test_df = load_input_data(opts.test_input)
    results_df = load_result_data(opts.results)
    determine_accuracy(test_df, results_df, "Candidates")