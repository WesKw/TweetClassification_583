import pandas as pd
import scipy

from time import time
from argparse import ArgumentParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
# from sklearn.pipeline import Pipeline


VALID_CLASSES=['-1', '0', '1']
STOP_WORDS_ONE=[]


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
    print(f"Time spent building classifier: {time() - start}s")

    return clf


def test_classifier(classifier, tweets, correct_classes, vectorizer, classifier_name="", print_evaluation=False):
    transformed_tweets = vectorizer.transform(tweets)
    prediction = classifier.predict(transformed_tweets)

    # print(prediction)
    if print_evaluation:
        precision = precision_score(correct_classes, prediction, labels=VALID_CLASSES, average=None)
        recall = recall_score(correct_classes, prediction, labels=VALID_CLASSES, average=None)
        f1score = f1_score(correct_classes, prediction, labels=VALID_CLASSES, average=None)
        accuracy = accuracy_score(correct_classes, prediction)

        print(f"Evaluation for {classifier_name}:")
        print(f"    Accuracy: {accuracy:.02f}")
        print(f"    Precision: [ Negative: {precision[0]:.02f} | Mixed: {precision[1]:.02f} | Positive: {precision[2]:.02f} ]")
        print(f"    Recall: [ Negative: {recall[0]:.02f} | Mixed: {recall[1]:.02f} | Positive: {recall[2]:.02f} ]")
        print(f"    F1 Score: [ Negative: {f1score[0]:.02f} | Mixed: {f1score[1]:.02f} | Positive: {f1score[2]:.02f} ]")

    return prediction


def clean_data(data: pd.DataFrame, ignore_class=False) -> pd.DataFrame:
    df = data.drop(index=0)
    df = df.set_axis(labels=['None', 'date', 'time', 'Anotated Tweet', 'Class', 'Yourclass'], axis=1) # fix labels
    df = df.drop(labels=['None', 'Yourclass'], axis=1) # drop irrelevant labels
    df["Class"] = df["Class"].astype("string")
    if not ignore_class:
        df = df[df["Class"].isin(VALID_CLASSES)] # remove any rows that have bad class labels
    df = df.dropna()

    # todo:: fix datetimes?

    # return cleaned df
    return df


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("training", help="The training data to use for tweet classification.")
    args.add_argument("test", help="The test data used to determine accuracy.")
    args.add_argument("--test-has-labels", action="store_true", help="If the test data has class labels, use these to calculate evaluation metrics.")
    args.add_argument("-o", "--output", help="The file to save the output to.")
    opts = args.parse_args()

    training_data = load_tweet_data(opts.training)
    obama_training_data = clean_data(training_data["Obama"], False).sample(frac=1).reset_index()
    romney_training_data = clean_data(training_data["Romney"], False).sample(frac=1).reset_index()

    # create our text vectorizers
    obama_text_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='ascii', max_df=0.5, min_df=0.0005, stop_words='english')
    romney_text_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='ascii', max_df=0.5, min_df=0.0005, stop_words='english')

    # build obama and romney classifiers
    obama_classifier = build_basic_classifier(obama_training_data, obama_text_vectorizer)
    romney_classifier = build_basic_classifier(romney_training_data, romney_text_vectorizer)

    # load test data
    test_data = load_tweet_data(opts.test)
    
    # load obama test data
    obama_test_data = clean_data(test_data["Obama"], ignore_class=(not opts.test_has_labels))
    obama_tweets = list(obama_test_data["Anotated Tweet"])
    obama_correct_classes = list(obama_test_data["Class"])

    # load romney test data
    romney_test_data = clean_data(test_data["Romney"], ignore_class=(not opts.test_has_labels))
    romney_tweets = list(romney_test_data["Anotated Tweet"])
    romney_correct_classes = list(romney_test_data["Class"])

    # test Obama
    obama_prediction = test_classifier(obama_classifier, obama_tweets, obama_correct_classes, obama_text_vectorizer, "Obama Classifier", opts.test_has_labels)
    obama_test_data["Your Class"] = obama_prediction
    print(obama_test_data)
    
    # test Romney
    romney_prediction = test_classifier(romney_classifier, romney_tweets, romney_correct_classes, romney_text_vectorizer, "Romney Classifier", opts.test_has_labels)
    romney_test_data["Your Class"] = romney_prediction
    print(romney_test_data)
