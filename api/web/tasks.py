import datetime
import os
import pickle
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.neural_network
import tensorflow
from celery import shared_task
from django.conf import settings
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, Input, LSTM, Bidirectional, Flatten
from keras.models import Sequential, Model
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from .models import Project, ProjectPrediction
from .models import ProjectFeature


@shared_task
def train_precipitation_prediction(project_id, pred_id):
    try:
        project_features = ProjectFeature.objects.filter(project_id=project_id)
        project = Project.objects.get(id=project_id)

        csv_delimiter = project.delimiter
        csv_file = os.path.join(settings.MEDIA_ROOT, project.dataset_file.name)
        file = open(csv_file, 'r')
        data = pd.read_csv(file, delimiter=csv_delimiter, parse_dates=['datetime'])
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

        def create_data_classification(df, columns, target, threshold):
            columns = columns[columns != target]

            df['avg'] = df[columns].mean(axis=1)

            df.loc[df[target] > df['avg'] * (1 + threshold), 'label'] = 0
            df.loc[df[target] <= df['avg'] * (1 + threshold), 'label'] = 1

            df = df.astype({'label': np.int})

            return df

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

        data.drop(skip_column_names, axis=1, inplace=True)

        data = create_data_classification(data, np.array(input_column_names), target_column_name, 0.3)

        X_train, X_test, y_train, y_test = split_df(data,
                                                    input_column_names,
                                                    target_column_name,
                                                    datetime.datetime(2020, 11, 1).date())

        clf = RandomForestClassifier(max_depth=7, n_estimators=250)
        clf.fit(X_train, y_train.values.ravel())
        # clf.fit(X_train, y_train)

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

        ProjectPrediction.objects.filter(id=pred_id).update(status=ProjectPrediction.StatusType.SUCCESS,
                                                            confusion_matrix=disp,
                                                            accuracy=accu,
                                                            precision=precision,
                                                            recall=recall,
                                                            f1_score=f1,
                                                            serialized_prediction_file=filename_model)
    except Exception as e:
        print(f'train_precipitation_prediction Error: {e}')
        ProjectPrediction.objects.filter(id=pred_id).update(status=ProjectPrediction.StatusType.ERROR)


@shared_task
def train_water_level_prediction(project_id, pred_id):
    @dataclass
    class KerasMLP:
        epochs = 5
        # epochs = 300
        layer_activation = 'relu'
        output_activation = 'softmax'
        optimizer = "adam"
        loss_function = 'categorical_crossentropy'
        batch_size = 50
        # batch_size = 10
        estimator = None
        model = None

        def default_algorithm(self, input_dim: int = None, out_dim: int = None):

            def compute_appends(ni, no):
                n = []

                while ni != no:

                    if no > int(ni / 2):
                        if ni not in n:
                            n.append(int(ni))
                        break

                    if ni % 2 == 0:
                        ni = ni / 2
                    else:
                        ni = (ni + 1) / 2

                    n.append(int(ni))

                if no in n:
                    n.remove(no)

                return n

            self.model = Sequential()
            self.model.add(Dense(input_dim, activation=self.layer_activation, input_dim=input_dim))

            ns = compute_appends(input_dim, out_dim)
            for n in ns:
                self.model.add(Dense(n, activation=self.layer_activation))
                # self.model.add(Dropout(0.2))

            self.model.add(Dense(out_dim, activation=self.output_activation))

            return self

        def compile_model(self):
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_function,
                metrics=['acc']
            )

        def train_model(self, x_train, y_train, x_train_validation=None, y_train_validation=None):

            # 1) Compile the model
            self.compile_model()

            # 2) Fit the model
            self.estimator = self.model.fit(
                x_train, y_train,
                validation_data=(x_train_validation, y_train_validation),
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=1,
                shuffle=False
            )

        def mlp_scikit(
                self,
                input_dim=None,
                out_dim=None,
                layer_activation=None,
                output_activaton=None
        ):
            self.model = sklearn.neural_network.MLPClassifier()

        def cnn(
                self,
                input_dim=None,
                out_dim=None,
                layer_activation=None,
                output_activaton=None
        ):
            # 1-D Convolutional Neural Network
            inputs = Input(shape=(input_dim, 1))
            x = Conv1D(32, 4, strides=1, padding='same', activation=layer_activation)(inputs)
            x = MaxPooling1D(pool_size=4)(x)
            x = Conv1D(64, 4, strides=1, padding='same', activation=layer_activation)(x)
            x = GlobalMaxPooling1D()(x)
            outputs = Dense(out_dim, activation=output_activaton)(x)
            self.model = Model(inputs=inputs, outputs=outputs, name='CNN')
            return self

        def lstm(
                self,
                input_dim=None,
                out_dim=None
        ):
            inputs = Input(shape=(input_dim, 1))
            x = Bidirectional(
                LSTM(64, return_sequences=True),
                merge_mode='concat')(inputs)
            x = Dropout(0.2)(x)
            x = Flatten()(x)
            outputs = Dense(out_dim, activation=self.output_activaton)(x)
            self.model = Model(inputs=inputs, outputs=outputs, name='LSTM')
            return self

    # -------- make x and y vectors -------------
    def make_vectors(
            data: pd.DataFrame = None,
            horizontal_ti: int = None,
            class_balancing: bool = True

    ):

        def combine_measured_and_anomalies(
                data: pd.DataFrame = None
        ):

            # mav = measured or anomalies values
            data["mav"] = np.where(
                pd.isna(data["anomaly_value"]),
                data["measured"],
                data["anomaly_value"]
            )

            # -------- mav labels ----------------
            # Start with all data labeled Normal (N)
            data["mav_labels"] = "N"

            # Spikes will get the S label
            # Stationary Values will get the SV label
            # Sensor Displacement will get the SD label
            anomaly_labels = {
                "Spikes": "S",
                "Stationary Values": "SV",
                "Sensor Displacement": "SD"
            }
            for anomaly_name in ["Spikes", "Stationary Values", "Sensor Displacement"]:
                data["mav_labels"] = np.where(
                    data["anomaly_name"] == anomaly_name,
                    anomaly_labels[anomaly_name],
                    data["mav_labels"]
                )
            # -------- mav labels ----------------

            return data

        def add_horizontal_scale(
                horizontal_ti: int = None,
                data: pd.DataFrame = None
        ):

            # ------------------------------ Time interval ------------------------------------------
            # The steps are:
            # 1) For each dt = 5 minutes add a date (shift_dt_d) to the dataframe, that is,
            # add the horizontal scale
            # 2) Where mav (the column with data measurements and anomalies values) == shift_dt_d add the
            # measure or anomaly value, that is, convert the dates in the horizontal scale into values

            # 1) For each dt = 5 minutes add a date (shift_dt_d) to the dataframe
            for dt in range(horizontal_ti):
                # Add the dates that represents the horizontal scale
                data[f"shift_{dt}_d"] = data["datetime"] - datetime.timedelta(minutes=5 * dt)

                # The code below is used to remove cases that do not
                # have enough measurements in the horizontal scale as examples of such cases
                # see the dates below
                # 2020-06-27T01:45:00Z -   2020-06-27T02:00:00Z =  15 minutes
                # 2020-07-15T05:30:00Z -   2020-07-15T05:50:00Z =  20 minutes
                # 2020-11-02T14:45:00Z -   2020-11-02T15:35:00Z =  50 minutes
                data[f"shift_{dt}_d"][
                    ~data[f"shift_{dt}_d"].isin(data["datetime"])
                ] = pd.NA

                # Not working
                # data = data[data[f"shift_{dt}_d"].notna()]

            # 2) Transform the dates from the horizontal scale into values
            for dt in range(horizontal_ti):
                data[f"shift_{dt}_v"] = np.where(
                    data[f"shift_{dt}_d"] == data["datetime"].shift(periods=dt),
                    data["mav"].shift(periods=dt),
                    pd.NA
                )

            # Remove the cases without enough measurements in the horizontal scale
            for dt in range(horizontal_ti):
                data = data[data[f"shift_{dt}_d"].notna()]
            # ---------------------------------------------------------------------------------------

            return data

        def make_class_balancing(
                number_of_samples: int = None,
                data: pd.DataFrame = None
        ):

            # ------------------------- balance the number of classes -------------------------------
            # classes can be a combination of [N, S, SV, SD]
            classes = data["mav_labels"].unique()

            classes_indexes = {}
            classes_indexes_balanced = {}
            for c in classes:
                # Get the indexes of the class c
                classes_indexes[c] = data.where(data["mav_labels"] == c)

                # Remove any case with nan
                classes_indexes[c] = classes_indexes[c].dropna(how='all')

                # make a list of the indexes
                classes_indexes[c] = list(classes_indexes[c].index)

                # Fo each class c a number_of_samples will be selected randomly
                classes_indexes_balanced[c] = []
                for i in range(number_of_samples):
                    # Randomly select a index for the class c
                    index = random.choice(classes_indexes[c])
                    classes_indexes_balanced[c].append(index)

            indexes_balanced = []
            for c in classes:
                indexes_balanced.extend(classes_indexes_balanced[c])

            # Cut the data for the selected indexes
            data = data.loc[indexes_balanced, :]

            # Reset the index
            data = data.reset_index()
            # --------------------------------------------------

            return data

        # Convert the string dates to datetime
        data["datetime"] = pd.to_datetime(
            data["datetime"],
            format="%Y-%m-%dT%H:%M:%SZ", errors='coerce'
        )

        # Add a mav (measured or anomalies values) columns in the dataframe
        # Add a mav labels N (Normal), S (Spikes), SV (Stationary Values), SD (Sensor Displacement)
        data = combine_measured_and_anomalies(data=data)

        # Add the columns related to the horizontal scale
        data = add_horizontal_scale(horizontal_ti=horizontal_ti, data=data)

        # Class balancing
        if class_balancing:
            data = make_class_balancing(
                number_of_samples=80,
                data=data
            )

        # ------------------------------------------ Make the y vector -----------------------------------------------------
        # Labels for classes
        # classes can be a combination of [N, S, SV, SD]
        classes = data["mav_labels"].unique()
        y_vector = label_binarize(data["mav_labels"], classes=classes)
        # ------------------------------------------------------------------------------------------------------------------

        # ------------------------------------------ Make the x vector -----------------------------------------------------
        train_columns = [f"shift_{dt}_v" for dt in range(horizontal_ti)]
        x_train = data[train_columns].astype(float)

        return x_train, y_vector

    try:
        # ------------------------------------------------
        # ----------- Global Parameters ------------------

        # Split Methodology = 1 => random data, 75% for training and 25% for validation
        # Split Methodology = 2 => ------------75%--------------- | --- 25% ---
        split_methodology = 1

        # hti = the horizontal scale in hours
        hti = 1
        horizontal_ti = hti * 12

        # Class balancing = True or False. If false then there will be a lot more "normal" data then anomalies and
        # processing time will be much higher.
        class_balancing = True

        project = Project.objects.get(id=project_id)
        csv_delimiter = project.delimiter
        csv_file = os.path.join(settings.MEDIA_ROOT, project.dataset_file.name)
        file = open(csv_file, 'r')
        data = pd.read_csv(file, delimiter=csv_delimiter, parse_dates=['datetime', 'updated_at'])
        path = "models/waterlevel/"
        model_base_path = os.path.join(settings.MEDIA_ROOT, path)

        x_vec, y_vec = make_vectors(
            data=data,
            horizontal_ti=horizontal_ti,
            class_balancing=class_balancing
        )
        # -------------------------------------------

        # ------ DataFrame Diff ----------------------
        # Procedure for the Stationary Values.
        # This procedure don't impact the other cases (S and SD).
        x_vec_temp = x_vec.reset_index(drop=True)
        x_vec_temp = x_vec_temp.diff(axis=1)
        x_vec[x_vec_temp == 0.] = -1.
        # ----------------------------------------------

        # -------------------- Split the vectors into x_train, x_est, y_train and y_test ---------------------------
        if split_methodology == 1:
            x_train, x_test, y_train, y_test = train_test_split(
                x_vec, y_vec, test_size=0.25
            )

        if split_methodology == 2:
            x_train, x_test, y_train, y_test = split_past_future_train_test(
                x_vec, y_vec, test_size=0.25
            )
        # ----------------------------------------------------------------------------------------------------------

        clf = KerasMLP().default_algorithm(
            input_dim=x_train.shape[1],
            out_dim=y_train.shape[1]
        )

        # Fit the data
        clf.train_model(
            x_train=x_train,
            y_train=y_train,
            x_train_validation=x_test,
            y_train_validation=y_test
        )

        predictions = clf.model.predict(x_test)
        confusion_matrix = tensorflow.math.confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
        confusion_matrix = np.array(confusion_matrix)
        confusion_matrix = confusion_matrix / confusion_matrix.sum()

        # For the Keras models
        model_name = f"KerasMLP_{clf.model.name}_{str(pred_id)}.h5"

        filename_model = f'{model_base_path}/{project_id}'

        def save_model_file(model, model_name, models_dir):
            """ Method that saves the model on local directory """

            # Checks if the directory of the models exists
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)

            # The full name = directory + name
            full_model_name = f'{models_dir}/{model_name}'

            # Save the new model
            model.save(full_model_name)

        # Save the models
        save_model_file(
            model_name=model_name, model=clf.model,
            models_dir=os.path.dirname(filename_model)
        )

        accu = clf.estimator.history['acc'][-1]
        # precision = metrics.precision_score(y_test, predictions)
        # recall = metrics.recall_score(y_test, predictions)
        # f1 = metrics.f1_score(y_test, predictions)

        output_file = f'{os.path.dirname(filename_model)}/{model_name}'
        ProjectPrediction.objects.filter(id=pred_id).update(status=ProjectPrediction.StatusType.SUCCESS,
                                                            confusion_matrix=confusion_matrix,
                                                            accuracy=accu,
                                                            serialized_prediction_file=output_file)
    except Exception as e:
        print(f'train_water_level_prediction Error: {e}')
        ProjectPrediction.objects.filter(id=pred_id).update(status=ProjectPrediction.StatusType.ERROR)
