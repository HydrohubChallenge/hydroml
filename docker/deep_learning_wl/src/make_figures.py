import numpy
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt


def confusion_matrix_figs(
        confusion_matrix: numpy.array = None,
        experiment_name: str = None,
        anomaly: str = None,
        figures_dir: pathlib.Path = None,
        model_name: str = None
):

    # Get the classes
    classes = ["N"]
    classes.extend([x.upper() for x in anomaly.split("+")])

    # ------- Fig 1: values
    # confusion_matrix = confusion_matrix / confusion_matrix.max()
    sns.reset_orig()
    plt.clf()
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.title('Confusion matrix ')
    cmt = numpy.array(confusion_matrix).T
    for x in range(confusion_matrix.shape[0]):
        for y in range(confusion_matrix.shape[1]):
            plt.text(x, y, str(round(cmt[x, y], 3)), horizontalalignment='center', verticalalignment='center')
    plt.colorbar()
    plt.savefig(figures_dir / f'{experiment_name}_{model_name}_confusion_matrix.png', dpi=80)

    # ------- Fig 2: normalized matrix
    confusion_matrix = confusion_matrix / confusion_matrix.sum()
    sns.reset_orig()
    plt.clf()
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.title('Confusion matrix ')
    cmt = numpy.array(confusion_matrix).T
    for x in range(confusion_matrix.shape[0]):
        for y in range(confusion_matrix.shape[1]):
            plt.text(x, y, str(round(cmt[x, y], 3)), horizontalalignment='center', verticalalignment='center')
    plt.colorbar()
    plt.savefig(figures_dir / f'{experiment_name}_{model_name}_confusion_matrix_normalized.png', dpi=80)


def accuracy_loss_figs(
        accuracy_train: list = None,
        accuracy_valid: list = None,
        loss_train: list = None,
        loss_valid: list = None,
        experiment_name: str = None,
        figures_dir: pathlib.Path = None,
        model_name: str = None
):

    sns.reset_orig()
    plt.clf()
    plt.plot(accuracy_train)
    plt.plot(accuracy_valid)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='lower right')
    plt.savefig(figures_dir / f'{experiment_name}_{model_name}_accuracy_valid.png', dpi=80)

    sns.reset_orig()
    plt.clf()
    plt.plot(loss_train)
    plt.plot(loss_valid)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.savefig(figures_dir / f'{experiment_name}_{model_name}_loss_valid.png', dpi=80)
