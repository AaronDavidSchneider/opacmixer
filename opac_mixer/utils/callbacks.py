"""Module with callbacks for training."""
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow import keras


def logerr(y_true, y_pred):
    """A logarithmic mean squared error."""
    mask = y_pred > 0
    try:
        return mean_squared_error(np.log(y_true[mask]), np.log(y_pred[mask]))
    except ValueError:
        return np.nan


def logr2(y_true, y_pred):
    """A r2 score for logarithmic data"""
    mask = y_pred > 0
    try:
        return r2_score(np.log(y_true[mask]), np.log(y_pred[mask]))
    except ValueError:
        return np.nan


class CustomCallback(keras.callbacks.Callback):
    """A custom callback for keras that prints out custom metrics"""

    def __init__(self, emulator, num_test=10000, errorfuncs=None):
        """
        Constructor of the callback class

        Parameters
        ----------
        emulator: Emulator
            the Emulator instance
        num_test: int
            The size of the test set used for validation
        errorfuncs (list(functions) or None):
            list of functions that are used for validation
        """
        super().__init__()
        if errorfuncs is None:
            errorfuncs = [logerr, logr2]
        self.validation_sets = []
        self._t_x = emulator.input_scaling
        self._ti_y = emulator.inv_output_scaling

        validation_sets = [
            (emulator.X_train, emulator.y_train),
            (emulator.X_test, emulator.y_test),
        ]
        self._validation_set_names = ["train", "test"]

        for X, y in validation_sets:
            valset = (
                X,
                y,
                np.random.randint(len(y), size=min([num_test, len(y)])),
            )
            self.validation_sets.append(valset)
        self.errorfuncs = errorfuncs

    def on_epoch_end(self, epoch, logs=None):
        """
        Callback for end of epoch

        Parameters
        ----------
        epoch: int
            the epoch
        logs: None or list
            The logs from keras which have the loss
        """
        errs = []
        for X_test, y_test, ti in self.validation_sets:
            y_pred = self._ti_y(
                X_test[ti],
                self.model.predict(self._t_x(X_test[ti]), verbose=0),
            )
            errs.append([err(y_test[ti], y_pred) for err in self.errorfuncs])

        val_err_str = (
            "("
            + "); (".join(
                [
                    f"{name} - "
                    + ", ".join(["{:.2e}".format(err) for err in errs_i])
                    for errs_i, name in zip(errs, self._validation_set_names)
                ]
            )
            + ")"
        )
        loss = logs["loss"]
        print(f"Epoch: {epoch}, loss: {loss:.2e}, val_error: {val_err_str}")
