import pandas as pd
import scipy

from time import time
from argparse import ArgumentParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
# from sklearn.pipeline import Pipeline


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

    return {"Obama": obama_df, "Romney": romney_df}


def build_basic_classifier(df: pd.DataFrame, text_vectorizer):
    start = time()
    tweets = list(df["Anotated Tweet"])
    classes = df["Class"]

    # use defaults for now
    # print(tweets)
    x_vector = text_vectorizer.fit_transform(tweets)
    clf = RidgeClassifier(solver="sparse_cg")
    clf.fit(x_vector, classes)
    print(time() - start)

    return clf


def test_classifier(clf, test_df, vectorizer):
    test_tweets = list(clean_data(test_data["Obama"])["Anotated Tweet"])
    transformed_tweets = vectorizer.transform(test_tweets)
    prediction = obama_classifier.predict(transformed_tweets)
    print(prediction)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data.drop(index=0)
    df = df.set_axis(labels=['None', 'date', 'time', 'Anotated Tweet', 'Class', 'Yourclass'], axis=1) # fix labels
    df = df.drop(labels=['None', 'Yourclass'], axis=1) # drop irrelevant labels
    df = df[(df["Class"] != 2) & (df["Class"] != "!!!!") & (df["Class"] != "irrevelant")] # remove any classes that are not 0, 1, -1
    df = df.dropna()
    df["Class"] = df["Class"].astype("string")

    # todo:: fix datetimes?

    # return cleaned df
    return df


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("training", help="The training data to use for tweet classification.")
    args.add_argument("test", help="The test data used to determine accuracy.")
    args.add_argument("-o", "--output", help="The file to save the output to.")
    opts = args.parse_args()

    training_data = load_tweet_data(opts.training)
    obama_data = clean_data(training_data["Obama"])
    romney_data = clean_data(training_data["Romney"])

    # create our text vectorizer
    text_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1, min_df=1, stop_words="english")

    # build obama and romney classifiers
    obama_classifier = build_basic_classifier(obama_data, text_vectorizer)

    test_data = load_tweet_data(opts.test)
    obama_test = list(clean_data(test_data["Obama"])["Anotated Tweet"])
    romney_test = clean_data(test_data["Romney"])

    # test Obama
    test_classifier(obama_classifier, test_data["Obama"], text_vectorizer)
    # transformed_tweets = text_vectorizer.transform(obama_test)
    # # transformed_obama = text_vectorizer.transform(obama_test)
    # # print(transformed_obama)
    # prediction = obama_classifier.predict(transformed_tweets)

    # print(prediction)

    # for key,test_data in test_dfs.items():
    #     test_data["df"] = clean_data(test_data["df"])
    #     print(test_data["df"])
        

    # load the test data after we train our model
    # test_dfs = load_tweet_data(opts.test)