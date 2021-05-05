import datetime
import json
import os
import pickle
import random
import ast
import sys
import zipfile
from os.path import basename

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from celery import shared_task
from django.conf import settings
from keras.layers import Dense
from keras.layers import LSTM as LSTM_keras
from keras.models import Sequential
from keras.models import model_from_json
from psycopg2.extensions import register_adapter, AsIs
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

from .models import Project, ProjectPrediction
from .models import ProjectFeature
from django.utils import timezone

def adapt_numpy_array(numpy_array):
    print(AsIs(tuple(numpy_array)))
    return AsIs(tuple(numpy_array))


register_adapter(np.ndarray, adapt_numpy_array)


@shared_task
def train_precipitation_prediction(project_id, pred_id):
    # try:
        project_features = ProjectFeature.objects.filter(project_id=project_id)
        project = Project.objects.get(id=project_id)
        pred_model = ProjectPrediction.objects.get(id=pred_id)
        params = ast.literal_eval(pred_model.parameters)
        csv_delimiter = project.delimiter
        csv_file = os.path.join(settings.MEDIA_ROOT, project.dataset_file.name)
        file = open(csv_file, 'r')
        data_train = pd.read_csv(file, delimiter=csv_delimiter, parse_dates=['datetime'])
        path = "models/precipitation/"
        model_base_path = os.path.join(settings.MEDIA_ROOT, path, str(project_id))

        def split_df(df, x_columns, y_column, day):
            df = df.reset_index()

            df_slice = df[df.datetime.dt.date < day]
            X_train = df_slice[x_columns]
            y_train = df_slice[y_column]
            df_slice = df[df.datetime.dt.date >= day]
            X_test = df_slice[x_columns]
            y_test = df_slice[y_column]
            return X_train, X_test, y_train, y_test

        input_column_names = []
        skip_column_names = []
        target_column_name = None
        timestamp_column_name = None

        for project_feature in project_features:
            if project_feature.type == ProjectFeature.Type.INPUT:
                input_column_names.append(project_feature.column)
            elif project_feature.type == ProjectFeature.Type.SKIP:
                skip_column_names.append(project_feature.column)
            elif project_feature.type == ProjectFeature.Type.TARGET:
                target_column_name = project_feature.column
            elif project_feature.type == ProjectFeature.Type.TIMESTAMP:
                timestamp_column_name = project_feature.column

        if len(skip_column_names) > 0:
            data_train.drop(skip_column_names, axis=1, inplace=True)

        X_train, X_test, y_train, y_test = split_df(data_train,
                                                    input_column_names,
                                                    target_column_name,
                                                    datetime.datetime(2020, 11, 1).date())

        # clf = RandomForestClassifier(max_depth=7, n_estimators=250)

        clf = RandomForestClassifier(max_depth=int(params['max_depth']), n_estimators=int(params['n_estimators']))

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accu = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)

        filename_model = f'{model_base_path}/{str(pred_id)}.pickle'
        model_dir = os.path.dirname(filename_model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        with open(filename_model, "wb") as f:
            pickle.dump(clf, f)

        disp = metrics.confusion_matrix(y_test, y_pred, normalize='true')
        disp = disp.round(decimals=4)

        array = []
        for item in disp:
            for i in item:
                array.append(i)

        input_column = json.dumps(input_column_names)
        target_column = json.dumps(target_column_name)
        timestamp_column = json.dumps(timestamp_column_name)
        skip_column = json.dumps(skip_column_names)

        ProjectPrediction.objects.filter(id=pred_id).update(status=ProjectPrediction.StatusType.SUCCESS,
                                                            confusion_matrix_array=array,
                                                            accuracy=accu,
                                                            precision=precision,
                                                            recall=recall,
                                                            f1_score=f1,
                                                            serialized_prediction_file=filename_model,
                                                            parameters=str(pred_model.parameters).translate(
                                                                {ord('{'): None, ord('}'): None, ord(','): '\n'}),
                                                            input_features=input_column.translate(
                                                                {ord('['): None, ord(']'): None, ord(','): '\n'}),
                                                            timestamp_features=timestamp_column,
                                                            skip_features=skip_column.translate(
                                                                {ord('['): None, ord(']'): None, ord(','): '\n'}),
                                                            target_features=target_column,
                                                            updated_at=datetime.datetime.now(tz=timezone.utc)
                                                            )
    # except Exception as e:
    #     print(f'train_precipitation_prediction Error: {e}')
    #     ProjectPrediction.objects.filter(id=pred_id).update(status=ProjectPrediction.StatusType.ERROR,
    #                                                         updated_at=datetime.datetime.now(tz=timezone.utc))


# Number of minutes to aggregate the data
MINUTES = 1

# Where the test data start
TEST_INIT_DATE = datetime.datetime(2020, 12, 1).date()

# Size of the look back
LOOK_BACK = 12

# The csv station ("santa_elena" or "hawkesworth_bridge")
STATION = "hawkesworth_bridge"

# Factor to multiply by MAPE in classification step
FACTOR = 10

# Set the random state to "1" to get reproducible results
def reset_random_seeds():

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ["PYTHONHASHSEED"] = str(1)

    # 2. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(1)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(1)

    # 4. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(1)

# Gets the "measured" time series and transform it in a supervised problem
# If the serie is: [1, 2, 3, 4, 5, 6] and the look_back is 3
# The data: [1, 2, 3], [2, 3, 4], [3, 4, 5]
# The labels: 4, 5 and 6
def create_supervised_data(original_data, look_back):

    new_data = original_data.copy()

    # Organize the data with look back
    # Features (remove first NaN columns)
    for shift in range(look_back, 0, -1):
        new_data[f"shift_{look_back - shift}"] = original_data["measured"].shift(periods=shift).iloc[look_back:]

    # Labels
    new_data["labels"] = original_data["measured"]
    new_data = new_data.iloc[look_back:]

    return new_data

# Get shift columns and labels column
def get_columns(look_back):
    x_columns = ["shift_{}".format(shift) for shift in range(look_back)]
    y_columns = "labels"

    return x_columns, y_columns

# MAPE calc of all predictions
def get_MAPE(predicted, expected):
    mape = np.mean(np.abs((expected - predicted) / expected))

    return (mape)

# Split the data into train and test set. Train: data before the "day". Test: data after the "day"
def split_dataset(df, day, look_back):

    # Get columns of interest
    x_columns, y_column = get_columns(look_back)
    df = df.reset_index()

    # Get train data
    df_slice = df[df.datetime.dt.date < day]
    X_train = df_slice[x_columns]
    y_train = df_slice[y_column]

    # Get test data
    df_slice = df[df.datetime.dt.date >= day]
    X_test = df_slice[x_columns]
    y_test = df_slice[y_column]

    print("  Train:")
    print("    - X_train: {}".format(X_train.shape))
    print("    - y_train: {}".format(y_train.shape))

    print("\n  Test:")
    print("    - X_test: {}".format(X_test.shape))
    print("    - y_test: {}".format(y_test.shape))

    # Put in a dictionary
    train_data = {"data": X_train, "labels": y_train}
    test_data = {"data": X_test, "labels": y_test}

    return train_data, test_data

# LSTM Model
class LSTM:

    # --------------------------------------------------------------------------------

    # Initialize the hyperparameters. Sets to default value if some param is "None"
    def init(self, look_back, n_samples, params=None):

        # To get reproductible results
        reset_random_seeds()

        # If params is None, set all the hyperparams to None (to get default values)
        if params is None:
            # Default:
            params = {
                # First Layer
                "units_1": 50,
                "activation_1": "sigmoid",
                "recurrent_activation_1": "relu",
                "use_bias_1": True,
                "unit_forget_bias_1": True,
                "dropout_1": 0.0,
                "recurrent_dropout_1": 0.0,
                "return_sequences_1": True,
                "return_state_1": False,
                "go_backwards_1": False,
                "stateful_1": 0.0,
                "unroll_1": False,

                # Second Layer
                "units_2": 50,
                "activation_2": "sigmoid",
                "recurrent_activation_2": "relu",
                "use_bias_2": True,
                "unit_forget_bias_2": True,
                "dropout_2": 0.0,
                "recurrent_dropout_2": 0.0,
                "return_sequences_2": False,
                "return_state_2": False,
                "go_backwards_2": False,
                "stateful_2": 0.0,
                "unroll_2": False,

                # General
                "epochs": 5,
                "batch_size": 1,
                "loss": "mean_squared_error",
                "optimizer": "adam"
            }

        # Model Hyperparameters (first layer)
        self.units_1 = params["units_1"]
        self.activation_1 = params["activation_1"]
        self.recurrent_activation_1 = params["recurrent_activation_1"]
        self.use_bias_1 = params["use_bias_1"]
        self.unit_forget_bias_1 = params["unit_forget_bias_1"]
        self.dropout_1 = params["dropout_1"]
        self.recurrent_dropout_1 = params["recurrent_dropout_1"]
        self.return_sequences_1 = params["return_sequences_1"]
        self.return_state_1 = params["return_state_1"]
        self.go_backwards_1 = params["go_backwards_1"]
        self.stateful_1 = params["stateful_1"]
        self.unroll_1 = params["unroll_1"]

        # Model Hyperparameters (second layer)
        self.units_2 = params["units_2"]
        self.activation_2 = params["activation_2"]
        self.recurrent_activation_2 = params["recurrent_activation_2"]
        self.use_bias_2 = params["use_bias_2"]
        self.unit_forget_bias_2 = params["unit_forget_bias_2"]
        self.dropout_2 = params["dropout_2"]
        self.recurrent_dropout_2 = params["recurrent_dropout_2"]
        self.return_sequences_2 = params["return_sequences_2"]
        self.return_state_2 = params["return_state_2"]
        self.go_backwards_2 = params["go_backwards_2"]
        self.stateful_2 = params["stateful_2"]
        self.unroll_2 = params["unroll_2"]

        # General
        self.epochs = params["epochs"]
        self.batch_size = params["batch_size"]
        self.loss = params["loss"]
        self.optimizer = params["optimizer"]

        # Dimensions
        self.n_timesteps = look_back
        self.n_features = 1
        self.n_outputs = 1
        self.n_samples = n_samples

        # Other params
        self.model = None
        self.model_fit = None

    # --------------------------------------------------------------------------------

    # Create and compile the model
    def build(self):

        # Create the model
        self.model = Sequential()

        # First Layer
        self.model.add(LSTM_keras(units=self.units_1,
                                  batch_input_shape=(self.batch_size, self.n_timesteps, self.n_features),
                                  activation=self.activation_1,
                                  recurrent_activation=self.recurrent_activation_1,
                                  use_bias=self.use_bias_1,
                                  unit_forget_bias=self.unit_forget_bias_1,
                                  dropout=self.dropout_1,
                                  recurrent_dropout=self.recurrent_dropout_1,
                                  return_sequences=self.return_sequences_1,
                                  return_state=self.return_state_1,
                                  go_backwards=self.go_backwards_1,
                                  stateful=self.stateful_1,
                                  unroll=self.unroll_1))

        # Second Layer
        self.model.add(LSTM_keras(units=self.units_2,
                                  activation=self.activation_2,
                                  recurrent_activation=self.recurrent_activation_2,
                                  use_bias=self.use_bias_2,
                                  unit_forget_bias=self.unit_forget_bias_2,
                                  dropout=self.dropout_2,
                                  recurrent_dropout=self.recurrent_dropout_2,
                                  return_sequences=self.return_sequences_2,
                                  return_state=self.return_state_2,
                                  go_backwards=self.go_backwards_2,
                                  stateful=self.stateful_2,
                                  unroll=self.unroll_2))

        # Output Layer
        self.model.add(Dense(self.n_outputs))

        # Compile the model
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    # --------------------------------------------------------------------------------

    # Flatten the data and train the model
    def train(self, train):

        # Flatten data
        train_data = train["data"].to_numpy()
        train_data = train_data.reshape(train_data.shape[0], self.n_timesteps, self.n_features)

        train_labels = train["labels"].to_numpy()
        train_labels = train_labels.reshape(train_labels.shape[0], 1)

        # Train
        self.model_fit = self.model.fit(train_data, train_labels, epochs=self.epochs,
                                        batch_size=self.batch_size, verbose=1)

    # --------------------------------------------------------------------------------

    # Flatten the data and predict one example
    def predict(self, example):

        # Flatten data
        data = np.array(example)
        data = data.reshape((1, data.shape[0], 1))

        # forescast the next week
        prediction = self.model.predict(data, verbose=0)

        # We only want the vector forecast
        prediction = float(prediction[0][0])

        return prediction

    # --------------------------------------------------------------------------------

    # Get the RMSE per week and the overall RMSE
    def evaluate_forecasts(self, expected, predicted):

        # RMSE per example
        rmses = []
        for i in range(expected.shape[0]):
            mse = mean_squared_error(np.asarray([expected[i]]), np.asarray([predicted[i]]))
            rmse = np.sqrt(mse)
            rmses.append(rmse)

        # Overall RMSE
        rmse_overall = expected - predicted
        rmse_overall = np.sqrt(np.mean(rmse_overall ** 2))

        return rmses, rmse_overall

    # --------------------------------------------------------------------------------

    # Train and predict the entire test set
    def evaluate_model(self, train, test):

        # Create and train the model
        self.train(train)

        # List of predictions
        predictions = list()

        for i in range(len(test["labels"])):
            print("\nExample {} / {}\n".format(i, len(test["labels"])))

            # Predict one day
            example = test["data"].iloc[i, :]
            prediction = self.predict(example)

            # Store the predictions
            predictions.append(prediction)

        # Evaluate predictions for each day and overall
        predicted = np.array(predictions)
        expected = np.array(test["labels"])

        scores, score = self.evaluate_forecasts(expected, predicted)

        results = {"predicted": predicted, "expected": expected, "overall_score": score, "daily_scores": scores}

        return results

    # --------------------------------------------------------------------------------

    # Predict the entire test set
    def evaluate_model_no_train(self, train, test):

        # List of predictions
        predictions = list()
        # --------------------------------------------------------------------------------

        # ------------------------------ Progress Bar --------------------------------

        initial_increment = int(len(test["label"]) // 100.0)
        increment = initial_increment

        sys.stdout.write('\r')
        sys.stdout.write("[%-100s] %d%%" % ('=' * 0, 0))
        sys.stdout.flush()

        j = 1
        for i in range(len(test["label"])):

            if i == increment:
                sys.stdout.write('\r')
                sys.stdout.write("[%-100s] %d%%" % ('=' * j, j))
                sys.stdout.flush()
                increment += initial_increment
                j += 1
            # ---------------------------------------------------------------------------

            # Predict one day
            example = test["data"].iloc[i, :]
            prediction = self.predict(example)

            # Store the predictions
            predictions.append(prediction)

        # Evaluate predictions for each day and overall
        predicted = np.array(predictions)
        expected = np.array(test["labels"])

        scores, score = self.evaluate_forecasts(expected, predicted)

        results = {"predicted": predicted, "expected": expected}

        return results

    # --------------------------------------------------------------------------------

    # Save the network
    def save(self, file_path, filename):
        filename_model = "{0}/{1}.json".format(file_path, filename)
        filename_weights = "{0}/{1}-weights.h5".format(file_path, filename)
        filename_zip = "{0}/{1}.zip".format(file_path, filename)

        # Save the model
        modelJson = self.model.to_json()
        with open(filename_model, "w") as jsonFile:
            jsonFile.write(modelJson)

        # Save the weights
        self.model.save_weights("{}".format(filename_weights))

        # Save the zip file to download
        zf = zipfile.ZipFile(filename_zip, mode='w')

        zf.write(filename_model, basename(filename_model))
        zf.write(filename_weights, basename(filename_weights))

        zf.close()

    # --------------------------------------------------------------------------------

    # Load the model and the weights
    def load(self, filename):

        # Load the json
        json_file = open("{}.json".format(filename, 'r'))
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)

        # Load the weights into the model
        loaded_model.load_weights("{}-weights.h5".format(filename))
        print("Loaded model from disk")

        return loaded_model

    # --------------------------------------------------------------------------------

    # Print the hyperparameters used
    def show_hyperparameters(self):
        print("LSTM Hyperparameters:")
        print("\n  First Layer:")
        print("    - Units: {}".format(self.units_1))
        print("    - Activation: {}".format(self.activation_1))
        print("    - Recurrent Activation: {}".format(self.recurrent_activation_1))
        print("    - Use Bias: {}".format(self.use_bias_1))
        print("    - Unit Forget Bias: {}".format(self.unit_forget_bias_1))
        print("    - Dropout: {}".format(self.dropout_1))
        print("    - Recurrent Dropout: {}".format(self.recurrent_dropout_1))
        print("    - Return Sequences: {}".format(self.return_sequences_1))
        print("    - Return State: {}".format(self.return_state_1))
        print("    - Go Backwards: {}".format(self.go_backwards_1))
        print("    - Stateful: {}".format(self.stateful_1))
        print("    - Unroll: {}".format(self.unroll_1))

        print("\n  Second Layer:")
        print("    - Units: {}".format(self.units_2))
        print("    - Activation: {}".format(self.activation_2))
        print("    - Recurrent Activation: {}".format(self.recurrent_activation_2))
        print("    - Use Bias: {}".format(self.use_bias_2))
        print("    - Unit Forget Bias: {}".format(self.unit_forget_bias_2))
        print("    - Dropout: {}".format(self.dropout_2))
        print("    - Recurrent Dropout: {}".format(self.recurrent_dropout_2))
        print("    - Return Sequences: {}".format(self.return_sequences_2))
        print("    - Return State: {}".format(self.return_state_2))
        print("    - Go Backwards: {}".format(self.go_backwards_2))
        print("    - Stateful: {}".format(self.stateful_2))
        print("    - Unroll: {}".format(self.unroll_2))

        print("\n  General:")
        print("    - Epochs: {}".format(self.epochs))
        print("    - Batch Size: {}".format(self.batch_size))
        print("    - Loss: {}".format(self.loss))
        print("    - Optimizer: {}\n".format(self.optimizer))

        print("\n  Dimensions:")
        print("    - Number of Timesteps: {}".format(self.n_timesteps))
        print("    - Number of Features: {}".format(self.n_features))
        print("    - Number of Outputs: {}\n".format(self.n_outputs))

    # --------------------------------------------------------------------------------

    # Show the dimensions of the network
    def show_architecture(self):
        print("\nNetwork Architecture:\n")
        self.model.summary()
        print("\n")

    # --------------------------------------------------------------------------------

    # Show the overall RMSE and the daily RMSE
    def summarize_scores(self, score, scores):

        print("\n\nLSTM Model Scores:\n")
        print("  - Overall RMSE: {:.4}\n".format(score))
        print("  - Daily RMSEs:\n")
        print("        day   rmse")
        for index, rmse in enumerate(scores):
            print("        {}{} => {:.4}".format("0" if index < 11 else "", index + 1, rmse))
    # --------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# After getting the regression, the next step is use a threshold based classifier
class Classifier:

    # --------------------------------------------------------------------------------

    exp_classes = None
    pred_classes = None
    mav = None
    misses = []

    # --------------------------------------------------------------------------------

    # Set some variables
    def init(self):
        self.exp_classes = None
        self.pred_classes = None
        self.mav = None
        self.misses = []

    # --------------------------------------------------------------------------------

    # Get mav data (measure + anomaly data)
    def get_mav_data(self, predicted, raw_data):

        # Size of test set
        last_test_values = predicted.shape[0]
        mav = np.where(pd.isna(raw_data["anomaly_value"]), raw_data["measured"], raw_data["anomaly_value"])
        mav = mav[-last_test_values:]
        self.mav = mav

        return mav

    # --------------------------------------------------------------------------------

    # Get the expected classes
    def get_expected_classes(self, expected, mav=None):

        # Stored inside the class
        if mav is None:
            mav = self.mav

        exp_classes = []
        for index, result in enumerate(expected):

            # If it's equal, than it is a normal data
            if result == mav[index]:
                exp_classes.append(0)

            # Otherwise, it's an anomaly data
            else:
                exp_classes.append(1)

        self.exp_classes = exp_classes

        return exp_classes

    # --------------------------------------------------------------------------------

    # Make the classification based on MAPE calc
    def get_classification(self, predicted, limit, mav=None):

        # Stored inside the class
        if mav is None:
            mav = self.mav

        classification = []

        for index, prediction in enumerate(predicted):
            if index >= len(mav):
                break

            # Calculates the MAPE

            mape = np.abs((mav[index] - prediction) / mav[index])

            # MAPE is greater than limit: it is an anomaly
            if mape > limit:

                # Indicates that the example is an anomaly
                classification.append(1)

            # Otherwise, it is a normal data
            else:

                # Indicates that the example is not an anomaly
                classification.append(0)

        self.pred_classes = classification

        return classification

    # --------------------------------------------------------------------------------

    # Calculate the accuracy, precision, recall and f1 score of the classification
    def get_metrics(self, anomaly_type, expected=None, predicted=None):

        if expected is None:
            expected = self.exp_classes

        if predicted is None:
            predicted = self.pred_classes

        TP = 0  # True Positive
        TN = 0  # True Negative
        FP = 0  # False Positive
        FN = 0  # False Negative

        classification = []

        for index, prediction in enumerate(predicted):

            # If it is right
            if expected[index] == prediction:

                # If it is an anomaly
                if prediction == 1:
                    TP += 1

                # If it is a normal data
                else:
                    TN += 1

            # Otherwise, it is wrong
            else:

                # Get the index where the error appear
                self.misses.append(index)

                # If it is an anomaly
                if prediction == 1:
                    FP += 1

                # If it is a normal data
                else:
                    FN += 1

        if anomaly_type == "s+sv+sd":
            print("Spike (S) and Stationary Value (SV) and Sensor Displacement (SD):")

        elif anomaly_type == "sv+sd":
            print("Stationary Value (SV) and Sensor Displacement (SD):")

        elif anomaly_type == "s+sd":
            print("Spike (S) and Sensor Displacement (SD):")

        elif anomaly_type == "s+sv":
            print("Spike (S) and Stationary Value (SV):")

        elif anomaly_type == "sv":
            print("Stationary Value (SV):")

        elif anomaly_type == "sd":
            print("Sensor Displacement (SD):")

        else:
            print("Spike (S):")

        print("\n  General:")
        print("    - TP: {}".format(TP))
        print("    - TN: {}".format(TN))
        print("    - FP: {}".format(FP))
        print("    - FN: {}".format(FN))
        print("    - Hits: {}".format(TP + TN))
        print("    - Misses: {}".format(FN + FP))
        print("    - Total (hits + misses): {}".format(TP + TN + FP + FN))

        total = TP + TN + FP + FN
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        confusion_matrix = [TN/total, FN/total, FP/total, TP/total]

        print("\n  Metrics:")
        print("    - Accuracy: {:.4}%".format(accuracy * 100.0))
        print("    - Precision: {:.4}%".format(precision * 100.0))
        print("    - Recall: {:.4}%".format(recall * 100.0))
        print("    - F1-Score: {:.4}%".format(f1_score * 100.0))

        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "confusion_matrix": confusion_matrix,
        }

        return metrics

    # --------------------------------------------------------------------------------

    # Function that plots the expected vs predicted vs mav values and draw a rectangle on anomalies (just FN)
    def show_the_wrongs_predictions(self, predicted, expected, all_datetime, miss_index, anomaly_type, mav=None):

        if len(self.misses) == 0:
            print("\n\nThe classification hit all the predictions\n\n")
            return

        if mav is None:
            mav = self.mav

        start = self.misses[miss_index] - 10
        end = self.misses[miss_index] + 10

        start = 0 if (start < 0) else start
        end = len(predicted) if (end > len(predicted)) else end

        # Get the datetime of test of
        x_axis = all_datetime[all_datetime.date >= TEST_INIT_DATE]
        x_axis = x_axis[start:end].strftime("%d %b %H:%M")

        # Defines the figure size
        sns.set(rc={'figure.figsize': (11, 10)})

        # Build the dataframes that will be plot
        d1 = pd.DataFrame({"Measured": np.asarray(expected[start:end])})
        d2 = pd.DataFrame({"Predicted": np.asarray(predicted[start:end])})
        d3 = pd.DataFrame({"Measured and Anomalies": np.asarray(mav[start:end])})

        # Creates the plot
        fig, ax = plt.subplots()
        ax.plot(x_axis, d1, label="Measured", marker="s")
        ax.plot(x_axis, d2, label="Predicted", marker="s")
        ax.plot(x_axis, d3, label="Measured and Anomalies", marker="s")

        # Draw the rectangles
        for index in range(start, end):

            # Gets the real measured and the anomaly value (from mav array)
            measure = np.asarray(expected)[index]
            anomaly = np.asarray(mav)[index]

            # If the measures are different, than an anomaly exists
            if measure != anomaly:

                # If the anomaly is above the measure on plot
                if anomaly > measure:

                    if anomaly_type == "sv":
                        padding = 0.01
                    else:
                        padding = 0.3

                    # Define the size of the rectangle (y axis)
                    top = abs(anomaly) + padding
                    bottom = measure - padding
                    size = top - bottom

                    # Creates the rectangle
                    if index == start + 10:
                        rect = patches.Rectangle((index - 1 - start, measure - padding), 2, size, linewidth=1,
                                                 edgecolor='r', facecolor='none')
                        ax.add_patch(rect)

                # If the anomaly is below the measure on plot
                else:

                    # Define the size of the rectangle (y axis)
                    top = measure + 0.3
                    bottom = anomaly - 0.3
                    size = top - bottom

                    # Creates the rectangle
                    if index == start + 10:
                        rect = patches.Rectangle((index - 1 - start, anomaly - 0.3), 2, size, linewidth=1,
                                                 edgecolor='r', facecolor='none')
                        ax.add_patch(rect)

        # Adds subtitle
        ax.legend()
        ax.set_ylabel("Measured")
        ax.set_xlabel("Time")

        plt.xticks(rotation=70)

        # Adds title and plot it
        plt.title("LSTM Model: Predicted VS Measured VS Measured and Anomalies")
        plt.show()
    # --------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------


@shared_task
def train_water_level_prediction(project_id, pred_id):
    try:
        project_features = ProjectFeature.objects.filter(project_id=project_id)
        project = Project.objects.get(id=project_id)
        pred_model = ProjectPrediction.objects.get(id=pred_id)
        params = ast.literal_eval(pred_model.parameters)
        csv_delimiter = project.delimiter
        csv_file = os.path.join(settings.MEDIA_ROOT, project.dataset_file.name)
        file = open(csv_file, 'r')
        data = pd.read_csv(file, delimiter=csv_delimiter, parse_dates=['datetime'])
        path = "models/water_level/"
        model_base_path = os.path.join(settings.MEDIA_ROOT, path, str(project_id))

        input_column_names = []
        skip_column_names = []
        target_column_name = None
        timestamp_column_name = None

        for project_feature in project_features:
            if project_feature.type == ProjectFeature.Type.INPUT:
                input_column_names.append(project_feature.column)
            elif project_feature.type == ProjectFeature.Type.SKIP:
                skip_column_names.append(project_feature.column)
            elif project_feature.type == ProjectFeature.Type.TARGET:
                target_column_name = project_feature.column
            elif project_feature.type == ProjectFeature.Type.TIMESTAMP:
                timestamp_column_name = project_feature.column

        if len(skip_column_names) > 0:
            data.drop(skip_column_names, axis=1, inplace=True)

        # Shift data using the "LOOK_BACK" (12)
        data_sup = create_supervised_data(data, LOOK_BACK)

        # Split data into train and test set
        train, test = split_dataset(data_sup, TEST_INIT_DATE, LOOK_BACK)

        # Best params found by the Talos Optimization
        best_params = {
            'units_1': 50,
            'activation_1': 'sigmoid',
            'recurrent_activation_1': 'relu',
            'use_bias_1': True,
            'unit_forget_bias_1': True,
            'dropout_1': 0.0,
            'recurrent_dropout_1': 0.0,
            'return_sequences_1': True,
            'return_state_1': False,
            'go_backwards_1': False,
            'stateful_1': False,
            'unroll_1': False,
            'units_2': 50,
            'activation_2': 'sigmoid',
            'recurrent_activation_2': 'relu',
            'use_bias_2': True,
            'unit_forget_bias_2': True,
            'dropout_2': 0.0,
            'recurrent_dropout_2': 0.0,
            'return_sequences_2': False,
            'return_state_2': False,
            'go_backwards_2': False,
            'stateful_2': False,
            'unroll_2': False,
            'optimizer': 'adam',
            'loss': 'mean_squared_error',
            'batch_size': int(params['batch_size']),
            'epochs': int(params['epochs']),
            'validation_split': 0.1
        }

        # Loads the trained model
        filename = "LSTM-measured-regression-model-{0}".format({str(pred_id)})
        file_path = os.path.join(model_base_path, filename)

        # Create, init, load and evaluate the LSTM
        # lstm = LSTM()
        # lstm.init(LOOK_BACK, len(train["labels"]), best_params)
        # lstm.model = lstm.load(model_base_path)
        # lstm.show_architecture()
        # results = lstm.evaluate_model_no_train(train, test)

        # If it needs to retrain
        lstm = LSTM()
        lstm.init(LOOK_BACK, len(train["labels"]), best_params)
        lstm.build()
        results = lstm.evaluate_model(train, test)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        lstm.save(file_path, filename)

        # Show the results
        # lstm.summarize_scores(results["overall_score"], results["daily_scores"])

        anomaly_type = "s+sv+sd"

        # Create and init the classifier
        clf = Classifier()
        clf.init()
        clf.exp_classes = data["label"].tolist()

        mape = np.mean(np.abs((results["expected"] - results["predicted"]) / results["expected"]))
        # Calculates the MAPE
        mape = get_MAPE(results["predicted"], results["expected"])

        # Defines the limit
        limit = mape * FACTOR
        print("\nLimit: {:.4}\n".format(limit))

        # Make the classification and show the metrics
        clf.get_classification(results["predicted"], limit, data["measured"])
        metrics = clf.get_metrics(anomaly_type)

        input_column = json.dumps(input_column_names)
        target_column = json.dumps(target_column_name)
        timestamp_column = json.dumps(timestamp_column_name)
        skip_column = json.dumps(skip_column_names)

        pred = []
        exp = []

        for i in results["predicted"]:
            pred.append(i)

        for i in results["expected"]:
            exp.append(i)

        ProjectPrediction.objects.filter(id=pred_id).update(status=ProjectPrediction.StatusType.SUCCESS,
                                                            predicted=pred,
                                                            expected=exp,
                                                            confusion_matrix_array=metrics['confusion_matrix'],
                                                            accuracy=metrics['accuracy'],
                                                            precision=metrics['precision'],
                                                            recall=metrics['recall'],
                                                            f1_score=metrics['f1_score'],
                                                            serialized_prediction_file=file_path,
                                                            parameters=str(pred_model.parameters).translate(
                                                                {ord('{'): None, ord('}'): None, ord(','): '\n'}),
                                                            input_features=input_column.translate(
                                                                {ord('['): None, ord(']'): None, ord(','): '\n'}),
                                                            timestamp_features=timestamp_column,
                                                            skip_features=skip_column.translate(
                                                                {ord('['): None, ord(']'): None, ord(','): '\n'}),
                                                            target_features=target_column,
                                                            updated_at=datetime.datetime.now(tz=timezone.utc)
                                                            )

    except Exception as e:
        print(f'train_water_level_prediction Error: {e}')
        ProjectPrediction.objects.filter(id=pred_id).update(status=ProjectPrediction.StatusType.ERROR,
                                                            updated_at=datetime.datetime.now(tz=timezone.utc))
