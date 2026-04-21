import pandas as pd
import tensorflow as tf
import re
import string
import requests
import numpy as np
import tensorflow_datasets as tfds
import torch

# from transformers import BertTokenizer, BertForSequenceClassification
from time import time
from argparse import ArgumentParser
from tensorflow.keras import layers
from tensorflow.keras import losses

VALID_CLASSES=['-1', '0', '1']

MAX_SEQUENCE_LENGTH=28
MAX_FEATURES=10000
SPLIT_SEED=42
BUFFER_SIZE=32
VECT_NAME="text_vectorization"
EPOCHS=10


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
    stripped = tf.strings.regex_replace(lower, r'<e>|<\/e>|#|[^a-zA-Z\d\s:]+|http.[\S]+', ' ')
    return tf.strings.regex_replace(stripped, '[%s]' % re.escape(string.punctuation), '')


def convert_pandas_df_to_tf_dataset(df: pd.DataFrame, batch_size=32, shuffle=True) -> tf.data.Dataset:
    # convert pandas dataframe to tf dataset
    df = df.drop(columns=["index", "date", "time"]).reset_index()
    # targets = df.pop("Class")
    dataset = tf.data.Dataset.from_tensor_slices((df["Anotated Tweet"].values, df["Class"].values))
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=BUFFER_SIZE, seed=SPLIT_SEED)
    return dataset


# this adapts the tensorflow tutorial on sentiment analysis for tweet data: https://www.tensorflow.org/tutorials/keras/text_classification
def do_learning_with_nn(data: pd.DataFrame, classifier_name: str, custom_test_data=None):
    longest_tweet = max(data["Anotated Tweet"].str.split(" ").str.len())
    print(f"Longest tweet has {longest_tweet} words.")
    max_sequence_length = longest_tweet

    # batch_size=BUFFER_SIZE
    batch_size=32

    dataset = convert_pandas_df_to_tf_dataset(data, batch_size)  
    split_data = tf.keras.utils.split_dataset(dataset, left_size=0.8, right_size=0.2, shuffle=True, seed=SPLIT_SEED)
    training_data = None
    testing_data = None

    # if we're using custom test data, convert it to a tf dataset and standardize.
    if custom_test_data is not None:
        training_data = split_data[0]
        validation_data = split_data[1]
        testing_data = convert_pandas_df_to_tf_dataset(custom_test_data, batch_size, False)
    else:
        # otherwise just use the split data
        training_data = split_data[0]
        testing_data = split_data[1]    
        # build the validation set like usual
        training_data,validation_data = tf.keras.utils.split_dataset(training_data, left_size=0.8, right_size=0.2, shuffle=False, seed=SPLIT_SEED)

    # create vectorization layer
    vectorizer = layers.TextVectorization(standardize=tweet_standardization, output_mode='int', output_sequence_length=max_sequence_length)
    # vectorizer = layers.TextVectorization(standardize=tweet_standardization, output_mode='tf_idf')

    train_text = training_data.map(lambda x, y: x)
    vectorizer.adapt(train_text)

    def vectorize_text(text, label):
        # text = tf.expand_dims(text, -1)
        return vectorizer(text), label
    
    text_batch, label_batch = next(iter(training_data))
    first_review, first_label = text_batch[0], label_batch[0]

    # apply vectorization to the datasets
    train_dataset = training_data.map(vectorize_text).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_data.map(vectorize_text).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = testing_data.map(vectorize_text).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # for x, y in train_dataset.take(1):
    #     print("Input shape:", x.shape)  # want (batch_size, max_sequence_length)
    #     print("Input dtype:", x.dtype)  # want int32 or int64
    #     print("Label shape:", y.shape)  # want (batch_size,)
    #     print("Unique labels:", np.unique(y.numpy()))  # want [0, 1, 2]
    #     print("Sample input:", x[0].numpy())  # want integers, not all 0s

    # vocab = vectorizer.get_vocabulary()
    # print("Vocab size:", len(vocab))         # should be close to MAX_FEATURES
    # print("Sample vocab:", vocab[:20]) 

    # labels = np.concatenate([y.numpy() for x, y in train_dataset])
    # print("Unique labels:", np.unique(labels))          # should be [0, 1, 2]
    # print("Label distribution:", np.bincount(labels))   # check class balance

    # for x, y in training_data.take(1):
    #     print("Raw text sample:", x[0].numpy())   # should be a real tweet string
    #     print("Raw label sample:", y[0].numpy())  # should be 0, 1, or 2

    # lengths = []
    # for x, y in train_dataset:
    #     # count non-zero tokens per sample
    #     lengths.extend(tf.math.count_nonzero(x, axis=1).numpy().tolist())

    # print("Average tweet length:", np.mean(lengths))   # probably very short
    # print("Max tweet length:", np.max(lengths))
    # print("95th percentile:", np.percentile(lengths, 95))

    # create neural network
    embedding_dim = 64
    model = tf.keras.Sequential([
        layers.Embedding(MAX_FEATURES+1, embedding_dim, mask_zero=True),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])

    # return None, None, None

    model.compile(loss=losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, callbacks=[callback])

    # model.fit(train_dataset.take(1), epochs=50)

    loss,accuracy = model.evaluate(test_dataset)
    print(f"Evaluating {classifier_name} classifier:")
    print(f"\tLoss={loss}")
    print(f"\tAccuracy={accuracy}")

    model.summary()

    return model,testing_data,vectorizer
    


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


def determine_performance_metrics(modified_df: pd.DataFrame):
    """
    Determines accuracy, precision, recall, and f1 score for a dataframe. The dataframe is expected to have a 
    "Class" column with the true class labels and a "Your Class" column with the predicted class labels.
    """
    def determine_accuracy(df: pd.DataFrame):
        correct = df[df["Class"] == df["Your Class"]]
        accuracy = len(correct) / len(df)
        print(f"Accuracy: {accuracy}")

    def determine_precision_recall_f1_for_class(df: pd.DataFrame, class_label: int):
        class_mapping = {
            1: "Positive",
            0: "Neutral",
            -1: "Negative"
        }

        true_positive = len(df[(df["Class"] == class_label) & (df["Your Class"] == class_label)])
        false_positive = len(df[(df["Class"] != class_label) & (df["Your Class"] == class_label)])
        false_negative = len(df[(df["Class"] == class_label) & (df["Your Class"] != class_label)])

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print("Metrics for class {}:".format(class_mapping.get(class_label, "Unknown")))
        print(f"\tPrecision: {precision}, Recall: {recall}, F1 Score: {f1_score}\n")

    determine_accuracy(modified_df)
    # determine performance metrics for positive and negative classes
    determine_precision_recall_f1_for_class(modified_df, 1)
    determine_precision_recall_f1_for_class(modified_df, -1)


def evaluate_bert_fined_tuned_model(model, test_data):
    ...


def evaluate_keras_model(model, test_data, vectorizer):
    # convert back to pandas df
    data = list(test_data.unbatch().as_numpy_iterator())
    test_df = pd.DataFrame(data, columns=["Anotated Tweet", "Class"])
    test_df["Anotated Tweet"] = test_df["Anotated Tweet"].apply(lambda x: x.decode("utf-8")).astype("str")

    print(test_df)

    # examples = tf.constant(test_data["Anotated Tweet"].tolist())
    result = model.predict(vectorizer(test_df["Anotated Tweet"].to_numpy()))
    print(len(result))
    tweet_class = []
    for data in result:
        tweet_class.append(np.argmax(data) - 1) # shift back the labels to match the original data

    # print(result)
    df = test_df
    df["Class"] = df["Class"] - 1 # shift back the labels to match the original data
    df["Your Class"] = tweet_class
    print(df)

    determine_performance_metrics(df)

    return df


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
    
    obama_model,obama_test_data,obama_vectorizer = do_learning_with_nn(obama_training, "Obama", testing_data)
    evaluate_keras_model(obama_model, obama_test_data, obama_vectorizer)

    # setup romney training,validation,test sets using the input data
    # romney_training = clean_data(training_data["Romney"], False).sample(frac=1).reset_index()
    # if testing_data is not None:
    #     testing_data = clean_data(testing_data["Romney"], False).sample(frac=1).reset_index()

    # romney_model,romney_test_data,romney_vectorizer = do_learning_with_nn(romney_training, "Romney", testing_data)
    # evaluate_keras_model(romney_model, romney_test_data, romney_vectorizer)
        
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
