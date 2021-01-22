import numpy
import tensorflow
from dataclasses import dataclass
import sklearn.neural_network
import pandas
import pathlib
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM, Bidirectional, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
import os

# ------------ Project Libs ----------------
from .make_figures import confusion_matrix_figs, accuracy_loss_figs
# ------------------------------------------


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


@dataclass
class KerasMLP:

    epochs = 5
    layer_activation = 'relu'
    output_activation = 'softmax'
    optimizer = "adam"
    loss_function = 'categorical_crossentropy'
    batch_size = 50
    estimator = None
    model = None

    def default_algorithm(self, input_dim: int = None, out_dim: int = None):

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

    def models_training_reports(
            self,
            figures_dir: pathlib.Path = None,
            experiment_name: str = None,
            x_test: pandas.DataFrame = None,
            y_test: pandas.DataFrame = None,
            anomaly: str = None,
            model_name: str = None
    ):

        with open(figures_dir / f'accuracy.txt', 'a') as file:
            file.write(
               f"{experiment_name} => "
               f"Training accuracy: {100 * self.estimator.history['acc'][-1]}, "
               f"Validation accuracy: {100 * self.estimator.history['val_acc'][-1]} "
               f"\n"
            )
        print(
            "Training accuracy: %.2f%% / Validation accuracy: %.2f%%" %
            (100 * self.estimator.history['acc'][-1], 100 * self.estimator.history['val_acc'][-1])
        )

        # -------------- Confusion matrix ---------------------------------------------------------------------
        predictions = self.model.predict(x_test)
        confusion_matrix = tensorflow.math.confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
        confusion_matrix = numpy.array(confusion_matrix)
        confusion_matrix_figs(
            confusion_matrix=confusion_matrix,
            anomaly=anomaly,
            experiment_name=experiment_name,
            figures_dir=figures_dir,
            model_name=model_name
        )
        # -----------------------------------------------------------------------------------------------------

        # -------------- Accuracy and Loss --------------------------------------------------------------------
        accuracy_loss_figs(
            accuracy_train=self.estimator.history['acc'],
            accuracy_valid=self.estimator.history['val_acc'],
            loss_train=self.estimator.history['loss'],
            loss_valid=self.estimator.history['val_loss'],
            experiment_name=experiment_name,
            figures_dir=figures_dir,
            model_name=model_name
        )
        # -----------------------------------------------------------------------------------------------------


def save_model_file(model, model_name, models_dir):
    """ Method that saves the model on local directory """

    # Checks if the directory of the models exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # The full name = directory + name
    full_model_name = models_dir / model_name

    # Remove the old models
    if os.path.exists(full_model_name):
        os.remove(full_model_name)

    # Save the new model
    model.save(full_model_name)
