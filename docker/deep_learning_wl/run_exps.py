import pathlib
import pandas
from sklearn.model_selection import train_test_split

# ------------ Project libs ---------
from src.train_test_vectors import (
    make_vectors,
    split_past_future_train_test
)
from src.models import KerasMLP, save_model_file
# ---------------------------------


# ---------------------------------
# ----------- Some global parameters ------------------
# The csvs names used in the experiments
csvs_names = {
        "s": "10k_S_anomalies.csv",
        "sv": "270_SV_anomalies.csv",
        "sd": "270_SD_anomalies.csv",
        "s+sv": "135_S_135_SV_anomalies.csv",
        "s+sd": "135_S_135_SD_anomalies.csv",
        "sv+sd": "135_SV_135_SD_anomalies.csv",
        "s+sv+sd": "80_S_80_SV_80_SD_anomalies.csv",
    }


def anomalies_experiment(
        anomaly: str = None,
        split_methodology: int = None,
        horizontal_ti: int = None,
        class_balancing: bool = True,
        results_dir: str = None,
        models_dir: str = None

):

    # Load the csv
    data = pandas.read_csv(f"./csvs/{csvs_names[anomaly]}")

    # -------- make x and y vectors -------------
    x_vec, y_vec = make_vectors(
        anomaly=anomaly,
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

    # The name of the experiment
    exp_name = f"Exp_Anomalies_{anomaly.replace('+', '_').upper()}_Horizontal_{hti}"

    # Name of the model
    model_name = f"{clf.model.name}"

    # Figs of train test
    clf.models_training_reports(
        figures_dir=pathlib.Path(results_dir),
        experiment_name=exp_name,
        x_test=x_test,
        y_test=y_test,
        anomaly=anomaly,
        model_name=model_name
    )

    # For the Keras models
    model_name = f"{exp_name}_KerasMLP_{clf.model.name}.h5"

    # Save the models
    save_model_file(
        model_name=model_name, model=clf.model,
        models_dir=pathlib.Path(models_dir)
    )


if __name__ == '__main__':

    # Split Methodology = 1 => random data, 75% for training and 25% for validation
    # Split Methodology = 2 => ------------75%--------------- | --- 25% ---
    split_methodology = 1

    # hti = the horizontal scale in hours
    hti = 1
    horizontal_ti = hti * 12
    # REMARK: hti * 12 because the data resolution time is 5 minutes, so 1 hour = 12 measurements
    # REMARK: for hti = 12 the number of measurements used to train the models will be 12 * 12 = 144
    # measurements, that is, a matrix with 144 columns! Such a large matrix will take time
    # to train the models (even in the balanced case)

    # Class balancing = True or False. If false then there will be a lot more "normal" data then anomalies and
    # processing time will be much higher.
    class_balancing = True

    # Directories
    results_dir = "./results/"
    models_dir = "./Models/"

    # Run S (Spikes) experiment
    anomalies_experiment(
        anomaly="s",
        split_methodology=split_methodology,
        horizontal_ti=horizontal_ti,
        class_balancing=class_balancing,
        results_dir=results_dir,
        models_dir=models_dir
    )

    # Run SV (Stationary Values) experiment
    anomalies_experiment(
        anomaly="sv",
        split_methodology=split_methodology,
        horizontal_ti=horizontal_ti,
        class_balancing=class_balancing,
        results_dir=results_dir,
        models_dir=models_dir
    )

    # Run SD (Sensor Displacement) experiment
    anomalies_experiment(
        anomaly="sd",
        split_methodology=split_methodology,
        horizontal_ti=horizontal_ti,
        class_balancing=class_balancing,
        results_dir=results_dir,
        models_dir=models_dir
    )
    # Run S+SV experiment
    anomalies_experiment(
        anomaly="s+sv",
        split_methodology=split_methodology,
        horizontal_ti=horizontal_ti,
        class_balancing=class_balancing,
        results_dir=results_dir,
        models_dir=models_dir
    )
    # Run S+SD experiment
    anomalies_experiment(
        anomaly="s+sd",
        split_methodology=split_methodology,
        horizontal_ti=horizontal_ti,
        class_balancing=class_balancing,
        results_dir=results_dir,
        models_dir=models_dir
    )

    # Run SV+SD experiment
    anomalies_experiment(
        anomaly="sv+sd",
        split_methodology=split_methodology,
        horizontal_ti=horizontal_ti,
        class_balancing=class_balancing,
        results_dir=results_dir,
        models_dir=models_dir
    )

    # Run S+SV+SD experiment
    anomalies_experiment(
        anomaly="s+sv+sd",
        split_methodology=split_methodology,
        horizontal_ti=horizontal_ti,
        class_balancing=class_balancing,
        results_dir=results_dir,
        models_dir=models_dir
    )

