import pandas as pd
import tensorflow as tf
import re
import string
import requests

from time import time
from argparse import ArgumentParser
from tensorflow.keras import layers
from tensorflow.keras import losses

VALID_CLASSES=['-1', '0', '1']

MAX_SEQUENCE_LENGTH=140 # tweets used to be maxed out at 140 characters
MAX_FEATURES=15000
SPLIT_SEED=42
BUFFER_SIZE=32


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


# standardize tweet text
def tweet_standardization(input):
    lower = tf.strings.lower(input)
    # remove punctuation and various oddities (hyperlinks)
    stripped = tf.strings.regex_replace(lower, r'<e>|<\/e>|#[\S]+|[^a-zA-Z\d\s:]+|http.[\S]+', '')
    return tf.strings.regex_replace(stripped, '[%s]' % re.escape(string.punctuation), '')


def convert_pandas_df_to_tf_dataset(df: pd.DataFrame, batch_size=32):
    # convert pandas dataframe to tf dataset
    df = df.drop(columns=["index", "date", "time"]).reset_index()
    # targets = df.pop("Class")
    dataset = tf.data.Dataset.from_tensor_slices((df["Anotated Tweet"].values, df["Class"].values))
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True).shuffle(buffer_size=batch_size)
    # dataset = dataset
    return dataset


# this adapts the tensorflow tutorial on sentiment analysis for tweet data: https://www.tensorflow.org/tutorials/keras/text_classification
def do_learning(data: pd.DataFrame, classifier_name: str, custom_test_data=None):
    batch_size=...

    dataset = convert_pandas_df_to_tf_dataset(data, BUFFER_SIZE)  
    split_data = tf.keras.utils.split_dataset(dataset, left_size=0.8, right_size=0.2, shuffle=True, seed=SPLIT_SEED)
    training_data = None
    testing_data = None

    # if we're using custom test data, convert it to a tf dataset and standardize.
    if custom_test_data is not None:
        training_data = split_data[0]
        validation_data = split_data[1]
        testing_data = convert_pandas_df_to_tf_dataset(custom_test_data, BUFFER_SIZE)
    else:
        # otherwise just use the split data
        training_data = split_data[0]
        testing_data = split_data[1]    
        # build the validation set like usual
        training_data,validation_data = tf.keras.utils.split_dataset(training_data, left_size=0.8, right_size=0.2, shuffle=True, seed=SPLIT_SEED)

    # create vectorization layer
    vectorizer = layers.TextVectorization(standardize=tweet_standardization, output_mode='int', output_sequence_length=MAX_SEQUENCE_LENGTH)
    # vectorizer = layers.TextVectorization(standardize=tweet_standardization, output_mode='tf_idf')

    train_text = training_data.map(lambda x, y: x)
    vectorizer.adapt(train_text)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorizer(text), label
    
    text_batch, label_batch = next(iter(training_data))
    first_review, first_label = text_batch[0], label_batch[0]

    # apply vectorization to the datasets
    train_dataset = training_data.map(vectorize_text).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_data.map(vectorize_text).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = testing_data.map(vectorize_text).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # create neural network
    embedding_dim = 16
    model = tf.keras.Sequential([
        layers.Embedding(MAX_FEATURES+1, embedding_dim),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model.summary()

    model.compile(loss=losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    epochs=25
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=[callback])

    loss,accuracy = model.evaluate(test_dataset)
    print(f"Evaluating {classifier_name} classifier:")
    print(f"\tLoss={loss}")
    print(f"\tAccuracy={accuracy}")

    model.summary()

    return model,test_dataset


def clean_data(data: pd.DataFrame, ignore_class=False) -> pd.DataFrame:
    df = data.drop(index=0)
    df = df.set_axis(labels=['None', 'date', 'time', 'Anotated Tweet', 'Class', 'Yourclass'], axis=1) # fix labels
    df = df.drop(labels=['None', 'Yourclass'], axis=1) # drop irrelevant labels
    df["Class"] = df["Class"].astype("str") # convert class labels to string for easier filtering
    if not ignore_class:
        df = df[df["Class"].isin(VALID_CLASSES)] # remove any rows that have bad class labels
    df["Class"] = df["Class"].astype("int32") # convert class labels back to int after filtering
    df["Class"] = df["Class"] + 1 # shift the labels by one to fit with tensorflow then we can subtract later.
    df = df.dropna()

    # standardize text data?

    # todo:: fix datetimes?

    # return cleaned df
    return df


def evaluate_model(model, test_data):
    examples = tf.constant(test_data["Anotated Tweet"].tolist())
    model.predict(examples)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("training", help="The training data to use for tweet classification.")
    args.add_argument("--custom-test", default=None, help="Test data used to evaluate the classifier. If not provided, the model will test against a fraction of the training data.")
    args.add_argument("--test-has-labels", action="store_true", help="If the test data has class labels, use these to calculate evaluation metrics.")
    args.add_argument("-o", "--output", help="The file to save the output to.")
    opts = args.parse_args()

    # load training data
    training_data = load_tweet_data(opts.training)
    testing_data = load_tweet_data(opts.custom_test) if opts.custom_test else None

    # setup obama training,validation,test sets using the input data
    obama_training = clean_data(training_data["Obama"], False).sample(frac=1).reset_index()
    if testing_data is not None:
        testing_data = clean_data(testing_data["Obama"], False).sample(frac=1).reset_index()
    
    model,test_data = do_learning(obama_training, "Obama", testing_data)
    evaluate_model(model, testing_data)

    # setup romney training,validation,test sets using the input data
    # romney_training = clean_data(training_data["Romney"], False).sample(frac=1).reset_index()
    # classifier = do_learning(romney_training)

    # create our text vectorizers
    # obama_text_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='ascii', max_df=0.5, min_df=0.0005, stop_words='english')
    # romney_text_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='ascii', max_df=0.5, min_df=0.0005, stop_words='english')

    # build obama and romney classifiers
    # obama_classifier = build_basic_classifier(obama_training_data, obama_text_vectorizer)
    # romney_classifier = build_basic_classifier(romney_training_data, romney_text_vectorizer)

    # load test data
    # test_data = load_tweet_data(opts.test)
    
    # load obama test data
    # obama_test_data = clean_data(test_data["Obama"], ignore_class=(not opts.test_has_labels))
    # obama_tweets = list(obama_test_data["Anotated Tweet"])
    # obama_correct_classes = list(obama_test_data["Class"])

    # load romney test data
    # romney_test_data = clean_data(test_data["Romney"], ignore_class=(not opts.test_has_labels))
    # romney_tweets = list(romney_test_data["Anotated Tweet"])
    # romney_correct_classes = list(romney_test_data["Class"])

    # test Obama
    # obama_prediction = test_classifier(obama_classifier, obama_tweets, obama_correct_classes, obama_text_vectorizer, "Obama Classifier", opts.test_has_labels)
    # obama_test_data["Your Class"] = obama_prediction
    # print(obama_test_data)
    
    # test Romney
    # romney_prediction = test_classifier(romney_classifier, romney_tweets, romney_correct_classes, romney_text_vectorizer, "Romney Classifier", opts.test_has_labels)
    # romney_test_data["Your Class"] = romney_prediction
    # print(romney_test_data)
