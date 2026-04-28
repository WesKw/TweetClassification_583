import pandas as pd
import re
import string
import numpy as np
import torch
import zipfile

from time import time
from argparse import ArgumentParser
from datasets import Dataset
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import TrainingArguments,Trainer

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
                f.write(f"({row['index']} {row['Your Class']})\n")
            f.write("))\n")
    
    zip_name = "WesleyKwiecinski.zip"
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        zipf.write(classifier_names[0] + ".txt")
        zipf.write(classifier_names[1] + ".txt")


def clean_data(data: pd.DataFrame, ignore_class=False) -> pd.DataFrame:
    df = data.drop(index=0)
    df = df.set_axis(labels=['None', 'date', 'time', 'Anotated Tweet', 'Class', 'Yourclass'], axis=1) # fix labels
    df = df.drop(labels=['None', 'date', 'time', 'Yourclass'], axis=1) # drop irrelevant labels
    df["Class"] = df["Class"].astype("str") # convert class labels to string for easier filtering
    if not ignore_class:
        df = df[df["Class"].isin(VALID_CLASSES)] # remove any rows that have bad class labels
    df["Class"] = df["Class"].astype("int32") # convert class labels back to int after filtering
    df["Class"] = df["Class"] + 1 # shift the labels by one to fit with tensorflow then we can subtract later.
    df = df.dropna()
    # print(df)

    # Even out class distribution

    # todo:: fix datetimes?

    # return cleaned df
    return df


def fine_tune_bert(data: pd.DataFrame, classifier_name: str, custom_test_data: bool = False, existing_trained: bool = False):



    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3) # positive, negative, neutral labels

    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)  # Must match your system CUDA
    
    training_set = None
    test_set = None
    validation_set = None
    df = data.set_axis(labels=["index", "Anotated Tweet", "labels"], axis=1)
    df["text"] = df["Anotated Tweet"].str.replace(STANDARDIZE_TWEET_REGEX, " ", regex=True).str.lower()

    # print(df)
    # print(custom_test_data)

    if custom_test_data is not None:
        # prep the custom test data if using, otherwise we just split the training data.
        test_set = custom_test_data.set_axis(labels=["index", "Anotated Tweet", "labels"], axis=1)
        test_set["text"] = test_set["Anotated Tweet"].str.replace(STANDARDIZE_TWEET_REGEX, " ", regex=True).str.lower()
        test_set = Dataset.from_pandas(test_set)
        training_set = df.sample(frac=0.8)
        validation_set = df.drop(training_set.index)
        training_set = Dataset.from_pandas(training_set)
        validation_set = Dataset.from_pandas(validation_set)
    else:
        training_set = df.sample(frac=0.8)
        test_set = df.drop(training_set.index)
        test_set = Dataset.from_pandas(test_set)

        validation_set = training_set.sample(frac=0.2)
        training_set = training_set.drop(validation_set.index)

        training_set = Dataset.from_pandas(training_set)
        validation_set = Dataset.from_pandas(validation_set)

    # print(training_set["text"][0])

    def preprocess_text(data):
        return tokenizer(
            data["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
        )
    
    tokenized_training = training_set.map(preprocess_text, batched=True)
    tokenized_validation = validation_set.map(preprocess_text, batched=True)
    tokenized_testing = test_set.map(preprocess_text, batched=True)

    # print(tokenized_training)
    # print(tokenized_training["Anotated Tweet"][0])
    # print(tokenized_training["input_ids"][0])
    # print(tokenized_training["attention_mask"][0])

    training_arguments = TrainingArguments(
        output_dir=f"./bert_{classifier_name}",
        num_train_epochs=7,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        logging_dir=f"./logs_{classifier_name}",
        logging_steps=5,
        save_strategy="best",
        load_best_model_at_end=True,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_training,
        eval_dataset=tokenized_validation
    )

    print("Fine-tuning existing BERT model...")
    trainer.train()
    model.eval()
    trainer.evaluate()

    return trainer,tokenized_testing


def determine_bert_performance_metrics(model, test_data):
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

    # sample_text = 
    output = model.predict(test_data)
    print(np.argmax(output.predictions, axis=-1))
    df = test_data.to_pandas()
    df["Your Class"] = np.argmax(output.predictions, axis=-1)
    df.drop(labels=["input_ids", "attention_mask", "token_type_ids"], axis=1, inplace=True)
    df.rename(columns={"text": "Anotated Tweet", "labels": "Class"}, inplace=True)
    df["Class"] = df["Class"].astype("int32") - 1 # shift back the labels to original format
    df["Your Class"] = df["Your Class"].astype("int32") - 1 # shift back the labels to original format
    df = df.sort_values(by="index").reset_index(drop=True)

    determine_accuracy(df)
    # # determine performance metrics for positive and negative classes
    determine_precision_recall_f1_for_class(df, 1)
    determine_precision_recall_f1_for_class(df, -1)

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


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("training", help="The training data to use for tweet classification.")
    args.add_argument("--custom-test", default=None, help="Test data used to evaluate the classifier. If not provided, the model will test against a fraction of the training data.")
    args.add_argument("--test-has-labels", action="store_true", help="If the test data has class labels, use these to calculate evaluation metrics.")
    args.add_argument("--force-binary", action="store_true", help="Force 2 separate binary classifiers for classifying tweet data.")
    args.add_argument("--method", action="append", help="Build a classifier using the given method.", choices=["multiclassifier", "multibinary", "bert"])
    args.add_argument("--save-test-results", action="store_true", help="Whether to save the test results to a file. Takes a file name. If not set, the test results will just be printed to the console.")
    args.add_argument("--save-bert-model", action="store_true", help="Whether to save the fine-tuned BERT model to a file named bert_model.pt")
    opts = args.parse_args()

    # load training data
    training_data = load_tweet_data(opts.training)
    testing_data = load_tweet_data(opts.custom_test) if opts.custom_test else None

    # setup obama training,validation,test sets using the input data
    obama_training = clean_data(training_data["Obama"], False).sample(frac=1).reset_index()
    obama_testing_data = None

    # setup Romney training,validation,test sets using the input data
    romney_training = clean_data(training_data["Romney"], False).sample(frac=1).reset_index()
    romney_testing_data = None

    if testing_data is not None:
        obama_testing_data = clean_data(testing_data["Obama"], False).sample(frac=1).reset_index()
        romney_testing_data = clean_data(testing_data["Romney"], False).sample(frac=1).reset_index()

    if "bert" in opts.method:
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        obama_model,obama_testing = fine_tune_bert(obama_training, "Obama", obama_testing_data)
        romney_model,romeny_testing = fine_tune_bert(romney_training, "Romney", romney_testing_data)

        obama_test_results = determine_bert_performance_metrics(obama_model, obama_testing)
        romney_test_results = determine_bert_performance_metrics(romney_model, romeny_testing)

        if opts.save_test_results:
            save_df_test_results([obama_test_results, romney_test_results], ["Obama", "Romney"], "results_bert")
