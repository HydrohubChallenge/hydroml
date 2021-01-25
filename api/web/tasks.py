from celery import shared_task
from .models import Project, ProjectPrediction
from django.conf import settings

import os
import pickle

import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics

import time

import matplotlib.pyplot as plt

@shared_task
def precipitation(project_id, pred_id):

    project_sel = Project.objects.get(id=project_id)

    csv_delimiter = project_sel.delimiter
    csv_file = os.path.join(settings.MEDIA_ROOT, project_sel.dataset.name)
    file = open(csv_file, 'r')
    data = pd.read_csv(file, delimiter=csv_delimiter, parse_dates= ['datetime'])
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

    data = create_data_classification(data, np.array(['hawkesworth_bridge']), 'santa_elena', 0.3)

    X_train, X_test, y_train, y_test = split_df(data,
                                                ['central_farm', 'chaa_creek', 'hawkesworth_bridge', 'santa_elena'],
                                                'label',
                                                datetime(2020, 11, 1).date())

    clf = RandomForestClassifier(max_depth=7, n_estimators=250)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accu = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    f1 = metrics.f1_score(y_test,y_pred)

    filename_model = f'{model_base_path}/{str(pred_id)}.pickle'
    model_dir = os.path.dirname(filename_model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(filename_model, "wb") as f:
        pickle.dump(clf, f)

    class_names = [0, 1]

    disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='true')

    # time.sleep(100)

    obj = ProjectPrediction.objects.get(id=pred_id)
    obj.status=True
    obj.confusion_matrix=disp.confusion_matrix
    obj.accuracy=accu
    obj.precision=precision
    obj.recall=recall
    obj.f1_score=f1
    obj.pickle=filename_model
    obj.save()

    return f1
