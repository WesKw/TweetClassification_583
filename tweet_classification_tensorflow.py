import pandas as pd
import tensorflow as tf
import re
import string
import numpy as np
import zipfile

from time import time
from argparse import ArgumentParser
from datasets import Dataset
from tensorflow.keras import layers
from tensorflow.keras import losses

VALID_CLASSES=['-1', '0', '1']

MAX_SEQUENCE_LENGTH=28 # 95th percentile of tweet length
MAX_FEATURES=10000
SPLIT_SEED=42
BUFFER_SIZE=32
VECT_NAME="text_vectorization"
EPOCHS=15

STANDARDIZE_TWEET_REGEX=r'<e>|<\/e>|#|[^a-zA-Z\d\s:]+|http.[\S]+'


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


def save_df_test_results(df_collection: list, classifier_names: list, output_file: str):
    """Save the results to an xlsx file, with each dataframe in the collection as a separate sheet."""
    # with pd.ExcelWriter(f"./{output_file}.xlsx") as writer:
    #     for df,classifier_name in zip(df_collection, classifier_names):
    #         df.to_excel(writer, sheet_name=f"{classifier_name}", index=False)

    for df,name in zip(df_collection, classifier_names):
        print("Outputting results to txt file")
        with open(name + ".txt", "w") as f:
            f.write("(setf x *(\n")
            for index,row in df.iterrows():
                f.write(f"({index} {row['Your Class']})\n")
            f.write("))\n")
    
    zip_name = "WesleyKwiecinski.zip"
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        zipf.write(classifier_names[0] + ".txt")
        zipf.write(classifier_names[1] + ".txt")


# standardize tweet text
def tweet_standardization(input):
    lower = tf.strings.lower(input)
    # remove punctuation and various oddities (hyperlinks)
    stripped = tf.strings.regex_replace(lower, STANDARDIZE_TWEET_REGEX, ' ')
    return tf.strings.regex_replace(stripped, '[%s]' % re.escape(string.punctuation), '')


def clean_data(data: pd.DataFrame, ignore_class=False, balance_dataset: bool=False) -> pd.DataFrame:
    df = data.drop(index=0)
    df = df.set_axis(labels=['None', 'date', 'time', 'Anotated Tweet', 'Class', 'Yourclass'], axis=1) # fix labels
    df = df.drop(labels=['None', 'date', 'time', 'Yourclass'], axis=1) # drop irrelevant labels
    df["Class"] = df["Class"].astype("str") # convert class labels to string for easier filtering
    if not ignore_class:
        df = df[df["Class"].isin(VALID_CLASSES)] # remove any rows that have bad class labels
    df["Class"] = df["Class"].astype("int32") # convert class labels back to int after filtering
    df["Class"] = df["Class"] + 1 # shift the labels by one to fit with tensorflow then we can subtract later.
    df = df.dropna()

    print(df)

    # balance the dataset by undersampling the majority class
    if balance_dataset:
        g = df.groupby("Class", group_keys=False)
        print(g)
        balanced_df = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()))).reset_index(drop=False).sort_values("index")
        balanced_df = balanced_df.join(df["Class"], on="index", how="left").reset_index(drop=True)
        print(balanced_df)
        df = balanced_df
        df.drop(columns=["index"], inplace=True)

    # Even out class distribution

    # todo:: fix datetimes?

    # return cleaned df
    return df


def convert_pandas_df_to_tf_dataset(df: pd.DataFrame, batch_size=32, shuffle=True, force_binary: bool = False, positive_label: int = 2) -> tf.data.Dataset:
    # convert pandas dataframe to tf dataset
    df = df.drop(columns=["index", "date", "time"]).reset_index()

    class_labels = df["Class"].values
    if force_binary:
        class_labels = df["Class"].apply(lambda x: 0 if x != positive_label else 1).values

    dataset = tf.data.Dataset.from_tensor_slices((df["Anotated Tweet"].values, class_labels))
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


def prep_data_for_multiple_binary(df: pd.DataFrame, batch_size=32, shuffle=True) -> pd.DataFrame:
    # convert pandas dataframe to tf dataset
    # tf_df = df.drop(columns=["index", "date", "time"]).reset_index()
    tf_df = df.reset_index()

    tf_df["BinaryPositive"] = tf_df["Class"].apply(lambda x: 0 if x != 2 else 1)
    tf_df["BinaryNegative"] = tf_df["Class"].apply(lambda x: 0 if x != 0 else 1)

    # dataset = tf.data.Dataset.from_tensor_slices((df["Anotated Tweet"].values, class_labels))
    # dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    # if shuffle:
    #     dataset = dataset.shuffle(buffer_size=BUFFER_SIZE, seed=SPLIT_SEED)
    return tf_df


def do_learning_multiple_binary(data: pd.DataFrame, classifier_name: str, custom_test_data=None, use_safe_mask=False):
    """
    We do a similar thing here, but instead of learning 1 model with 3 classes, we do 2 separate binary classifiers, one 
    for positive / not-positive and one for negative / not-negative. Then for testing, we apply both models. 
    If one model produces positive, and another produces negative, then we say the review is mixed. Or, if they both say
    neither, then we also say its mixed.
    """
    class SafeGlobalAveragePooling1D(tf.keras.layers.Layer):
        def call(self, inputs, mask=None):
            if mask is not None:
                mask = tf.cast(mask, inputs.dtype)
                mask = tf.expand_dims(mask, -1)
                inputs *= mask
                return tf.reduce_sum(inputs, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-8)
            return tf.reduce_mean(inputs, axis=1)
        
    tf_df = prep_data_for_multiple_binary(data) # we need to keep the testing data and split and convert when we actually evaluate the models
    training_data = None
    test_df = None

    if custom_test_data is not None:
        # prep the custom test data if using, otherwise we just split the training data.
        test_df = prep_data_for_multiple_binary(custom_test_data)
    else:
        training_data = tf_df.sample(frac=0.8)
        test_df = tf_df.drop(training_data.index)

    def build_binary_model_for_class(data: pd.DataFrame, classifier_name: str, class_label: int):
        # batch_size=BUFFER_SIZE
        batch_size=32

        """2 is positive, 1 is mixed, 0 is negative"""
        initial_tf_dataset = tf.data.Dataset.from_tensor_slices((data["Anotated Tweet"].values, (data["BinaryPositive"].values if  class_label == 2 else data["BinaryNegative"].values)))
        
        # print(initial_tf_dataset)
        initial_tf_dataset = initial_tf_dataset.filter(lambda x,y: tf.reduce_any(tf.strings.length(x) > 0))
        
        training,validation = tf.keras.utils.split_dataset(
            initial_tf_dataset.batch(batch_size=batch_size, drop_remainder=True).shuffle(buffer_size=BUFFER_SIZE, seed=SPLIT_SEED),
            left_size = 0.8,
            right_size = 0.2,
            shuffle=False,
            seed=SPLIT_SEED
        )

        # create vectorization layer
        vectorizer = layers.TextVectorization(standardize=tweet_standardization, output_mode='int', output_sequence_length=MAX_SEQUENCE_LENGTH)

        train_text = training.map(lambda x, y: x)
        vectorizer.adapt(train_text)

        def vectorize_text(text, label):
            text = tf.expand_dims(text, -1)
            return vectorizer(text), label

        # apply vectorization to the datasets
        train_dataset = training.map(vectorize_text).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        validation_dataset = validation.map(vectorize_text).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        # test_dataset = testing_data.map(vectorize_text).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        for x,y in train_dataset.take(-1):
            if x[0].numpy().sum() == 0:
                print(x[0])

        mask_layer = layers.GlobalAveragePooling1D if not use_safe_mask else SafeGlobalAveragePooling1D

        # tf.debugging.enable_check_numerics()
        # create neural network
        embedding_dim = 16
        model = tf.keras.Sequential([
            layers.Embedding(MAX_FEATURES+1, embedding_dim, mask_zero=True),
            layers.Dropout(0.2),
            mask_layer(),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(loss=losses.BinaryCrossentropy(), optimizer="adam", metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)])
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, callbacks=[callback])
        model.summary()

        return model,vectorizer
    
    pos_model,vectorizer = build_binary_model_for_class(tf_df, classifier_name, class_label=2)
    neg_model,vectorizer = build_binary_model_for_class(tf_df, classifier_name, class_label=0)

    return (pos_model,vectorizer),(neg_model,vectorizer),test_df


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
    # print(df)

    determine_performance_metrics(df)

    return df


def determine_performance_metrics_multibinary(positive_model, negative_model, testing_data, classifier_label):
    """
    Determines accuracy, precision, recall, and f1 score for a dataframe. The dataframe is expected to have a 
    "Class" column with the true class labels and a "Your Class" column with the predicted class labels.
    """
    pos_mod = positive_model[0]
    pos_vect = positive_model[1]
    neg_mod = negative_model[0]
    neg_vect = negative_model[1]

    positive_result = pos_mod.predict(pos_vect(testing_data["Anotated Tweet"].to_numpy()))
    negative_result = neg_mod.predict(neg_vect(testing_data["Anotated Tweet"].to_numpy()))
    predicted_classes = []

    for pos_pred,neg_pred in zip(positive_result, negative_result):
        is_pos = False
        is_neg = False
        is_mix = False

        if pos_pred >= 0.5:
            is_pos = True

        if neg_pred >= 0.5:
            is_neg = True

        if is_pos and is_neg or (not is_pos and not is_neg):
            is_mix = True
            is_pos = False
            is_neg = False

        if is_mix:
            predicted_classes.append(0)
        elif is_pos:
            predicted_classes.append(1)
        elif is_neg:
            predicted_classes.append(-1)

    modified_df = testing_data
    modified_df["Class"] = modified_df["Class"] - 1
    modified_df["Your Class"] = predicted_classes

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

        print("\tMetrics for class {}:".format(class_mapping.get(class_label, "Unknown")))
        print(f"\t\tPrecision: {precision}, Recall: {recall}, F1 Score: {f1_score}\n")

    print(f"Evaluating {classifier_label} classifier:")
    determine_accuracy(modified_df)
    # determine performance metrics for positive and negative classes
    determine_precision_recall_f1_for_class(modified_df, 1)
    determine_precision_recall_f1_for_class(modified_df, -1)

    return modified_df


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("training", help="The training data to use for tweet classification.")
    args.add_argument("--custom-test", default=None, help="Test data used to evaluate the classifier. If not provided, the model will test against a fraction of the training data.")
    args.add_argument("--test-has-labels", action="store_true", help="If the test data has class labels, use these to calculate evaluation metrics.")
    args.add_argument("--force-binary", action="store_true", help="Force 2 separate binary classifiers for classifying tweet data.")
    args.add_argument("--method", action="append", help="Build a classifier using the given method.", choices=["multiclassifier", "multibinary", "bert"])
    args.add_argument("--save-test-results", action="store_true", help="Whether to save the test results to a file. Takes a file name. If not set, the test results will just be printed to the console.")
    args.add_argument("--balance-dataset", action="store_true", help="Whether to balance the dataset by undersampling the majority class.")
    opts = args.parse_args()

    # load training data
    training_data = load_tweet_data(opts.training)
    testing_data = load_tweet_data(opts.custom_test) if opts.custom_test else None

    # setup obama training,validation,test sets using the input data
    obama_training = clean_data(training_data["Obama"], False, opts.balance_dataset).sample(frac=1).reset_index()
    obama_testing_data = None

    # setup Romney training,validation,test sets using the input data
    romney_training = clean_data(training_data["Romney"], False, opts.balance_dataset).sample(frac=1).reset_index()
    romney_testing_data = None

    if testing_data is not None:
        obama_testing_data = clean_data(testing_data["Obama"], False).sample(frac=1).reset_index()
        romney_testing_data = clean_data(testing_data["Romney"], False).sample(frac=1).reset_index()

    if not opts.method:
        print("Nothing to do.")
        exit()

    if "multibinary" in opts.method:
        print("Building and testing multiple binary classifiers...")
        obama_pos_mod,obama_neg_mod,obama_test_df = do_learning_multiple_binary(obama_training, "Obama", obama_testing_data, use_safe_mask=False)
        romney_pos_mod,romney_neg_mod,romney_test_df = do_learning_multiple_binary(romney_training, "Romney", romney_testing_data, use_safe_mask=True)

        obama_test_results = determine_performance_metrics_multibinary(obama_pos_mod, obama_neg_mod, obama_test_df, "Obama")        
        romney_test_results = determine_performance_metrics_multibinary(romney_pos_mod, romney_neg_mod, romney_test_df, "Romney")

        if opts.save_test_results:
            save_df_test_results([obama_test_results, romney_test_results], ["Obama", "Romney"], "results_multibinary")

    if "multiclassifier" in opts.method:
        print("Building and testing 1 multi-class classifier...")
        obama_model,obama_test_data,obama_vectorizer = do_learning_with_nn(obama_training, "Obama", obama_testing_data)
        romney_model,romney_test_data,romney_vectorizer = do_learning_with_nn(romney_training, "Romney", romney_testing_data)

        obama_test_results = evaluate_keras_model(obama_model, obama_test_data, obama_vectorizer)
        romney_test_results = evaluate_keras_model(romney_model, romney_test_data, romney_vectorizer)

        if opts.save_test_results:
            save_df_test_results([obama_test_results, romney_test_results], ["Obama", "Romney"], "results_single")
