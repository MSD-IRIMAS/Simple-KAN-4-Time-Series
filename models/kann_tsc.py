from kan import KAN
import torch
import numpy as np

from aeon.transformations.collection.feature_based import Catch22

from typing import Tuple


class KAN_CLASSIFIER:
    def __init__(
        self,
        width: list,
        output_dir: str = "./",
        steps: int = 20,
        k: int = 3,
        grid: int = 5,
        random_state: int = None,
    ):
        """KAN Time Series Classifier.

        Parameters
        ----------
        width: list
            The width of the KAN layers.
        output_dir: str, default = "./"
            The output directory.
        steps: int, default = 20
            The number of optimization steps.
        k: int, default = 3
            The order of piecewise polynomial.
        grid: int, default = 5
            The number of grid intervals.
        random_state: int, default = None
            The random state for the inital seed.
        """
        self.width = width
        self.output_dir = output_dir
        self.steps = steps
        self.k = k
        self.grid = grid
        self.random_state = random_state

    def _train_acc(
        self,
    ):
        return torch.mean(
            (
                torch.argmax(self.model(self.dataset["train_input"]), dim=1)
                == self.dataset["train_label"]
            ).float()
        )

    def _test_acc(
        self,
    ):
        return torch.mean(
            (
                torch.argmax(self.model(self.dataset["test_input"]), dim=1)
                == self.dataset["test_label"]
            ).float()
        )

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
        xtrain = catch22_transformer.fit_transform(np.expand_dims(xtrain, axis=1))
        xval = catch22_transformer.fit_transform(np.expand_dims(xval, axis=1))

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

        self.dataset = {
            "train_input": torch.from_numpy(xtrain).to(self.device),
            "train_label": torch.from_numpy(ytrain).to(self.device),
            "test_input": torch.from_numpy(xval).to(self.device),
            "test_label": torch.from_numpy(yval).to(self.device),
        }

        assert int(xtrain.shape[1]) == int(xval.shape[1])
        assert len(xtrain.shape) == len(xval.shape) == 2
        assert len(xtrain) == len(ytrain)
        assert len(xval) == len(yval)

        self.length_TS = int(xtrain.shape[1])
        self.n_classes = len(np.unique(ytrain))

        if self.width is not None:
            self.width_ = [22] + self.width + [self.n_classes]
        else:
            self.width_ = [22, self.n_classes]

        self.model = KAN(
            width=self.width_,
            grid=self.grid,
            k=self.k,
            seed=self.random_state,
            device=self.device,
        )

        self.model.to(self.device)

        self.results = self.model.train(
            self.dataset,
            opt="LBFGS",
            steps=self.steps,
            metrics=(self._train_acc, self._test_acc),
            loss_fn=torch.nn.CrossEntropyLoss(),
            device=self.device,
        )

        return self.results["_train_acc"][-1], self.results["_test_acc"][-1]

    def get_symbolic_function(
        self,
    ):
        lib = [
            "x",
            "x^2",
            "x^3",
            "x^4",
            "exp",
            "log",
            "sqrt",
            "tanh",
            "sin",
            "tan",
            "abs",
        ]
        self.model.auto_symbolic(lib=lib)
        self.formulas = self.model.symbolic_formula()[0]

        return [_formula for _formula in self.formulas]
