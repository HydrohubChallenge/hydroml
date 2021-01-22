import pandas
import numpy
import datetime
import random
from sklearn.preprocessing import label_binarize
import tensorflow

# The number of anomalies samples
number_of_anomalies_samples = {
        "s": 10000,
        "sv": 270,
        "sd": 270,
        "s+sv": 135,
        "s+sd": 135,
        "sv+sd": 135,
        "s+sv+sd": 80,
    }


# TODO: turn this code into a class method
def combine_measured_and_anomalies(
        data: pandas.DataFrame = None
):

    # mav = measured or anomalies values
    data["mav"] = numpy.where(
        pandas.isna(data["anomaly_value"]),
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
        data["mav_labels"] = numpy.where(
            data["anomaly_name"] == anomaly_name,
            anomaly_labels[anomaly_name],
            data["mav_labels"]
        )
    # -------- mav labels ----------------

    return data


# TODO: turn this code into a class method
def add_horizontal_scale(
        horizontal_ti: int = None,
        data: pandas.DataFrame = None
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
        ] = pandas.NA

        # Not working
        # data = data[data[f"shift_{dt}_d"].notna()]

    # 2) Transform the dates from the horizontal scale into values
    for dt in range(horizontal_ti):
        data[f"shift_{dt}_v"] = numpy.where(
            data[f"shift_{dt}_d"] == data["datetime"].shift(periods=dt),
            data["mav"].shift(periods=dt),
            pandas.NA
        )

    # Remove the cases without enough measurements in the horizontal scale
    for dt in range(horizontal_ti):
        data = data[data[f"shift_{dt}_d"].notna()]
    # ---------------------------------------------------------------------------------------

    return data


# TODO: turn this code into a class method
def make_class_balancing(
        number_of_samples: int = None,
        data: pandas.DataFrame = None
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

def split_past_future_train_test(x_vec, y_vec, test_size=0.25):

    x_vec = x_vec.reset_index(drop=True)

    size = x_vec.shape[0]
    train_size = int((1 - test_size) * size)

    x_train = x_vec.loc[:train_size-1, :]
    x_test = x_vec.loc[train_size:size, :]

    y_train = y_vec[:train_size, :]
    y_test = y_vec[train_size:size, :]

    return x_train, x_test, y_train, y_test


def make_vectors(
        anomaly: str = None,
        data: pandas.DataFrame = None,
        horizontal_ti: int = None,
        class_balancing: bool = True

):

    # Convert the string dates to datetime
    data["datetime"] = pandas.to_datetime(
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
            number_of_samples=number_of_anomalies_samples[anomaly],
            data=data
        )

    # ------------------------------------------ Make the y vector -----------------------------------------------------
    # Labels for classes
    # classes can be a combination of [N, S, SV, SD]
    classes = data["mav_labels"].unique()
    y_vector = label_binarize(data["mav_labels"], classes=classes)
    if anomaly in ["s", "sv", "sd"]:
        y_vector = tensorflow.keras.utils.to_categorical(y_vector, num_classes=len(classes))
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------ Make the x vector -----------------------------------------------------
    train_columns = [f"shift_{dt}_v" for dt in range(horizontal_ti)]
    x_train = data[train_columns].astype(float)

    return x_train, y_vector
