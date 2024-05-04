import numpy as np

from aeon.transformations.collection.feature_based import Catch22

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from typing import Tuple

class LP_CLASSIFIER:
    def __init__(self, random_state: int = None) -> None:
        """LP Time Series Classifier.

        Parameters
        ----------
        random_state: int, default = None
            The random state for the inital seed.
        """
        self.random_state = random_state

    def fit_and_validate(
        self, xtrain: np.ndarray, ytrain: np.ndarray, xval: np.ndarray, yval: np.ndarray
    ) -> Tuple[float, float]:
        """Training and Evaluating the model.
        
        Parameters
        ----------
        xtrain: np.ndarray of shape (n_instances, n_timepoints)
            The input time series for training.
        ytrain: np.ndarray of shape (n_instances)
            The labels of the training samples
        xval: np.ndarray of shape (n_instances, n_timepoints)
            The input time series for validation.
        yval: np.ndarray of shape (n_instances)
            The labels of the validation samples
            
        Returns
        -------
        Tuple[float, float]
            The accuracies on both train and validation.
        """
        catch22_transformer = Catch22(use_pycatch22=True)
        xtrain_transformed = catch22_transformer.fit_transform(
            np.expand_dims(xtrain, axis=1)
        )
        xval_transformed = catch22_transformer.transform(np.expand_dims(xval, axis=1))

        lr_classifier = LogisticRegression(
            penalty="none", multi_class="multinomial", random_state=self.random_state
        )

        lr_classifier.fit(xtrain_transformed, ytrain)

        ypred_train = lr_classifier.predict(xtrain_transformed)
        ypred_test = lr_classifier.predict(xval_transformed)

        return accuracy_score(
            y_true=ytrain, y_pred=ypred_train, normalize=True
        ), accuracy_score(y_true=yval, y_pred=ypred_test, normalize=True)
