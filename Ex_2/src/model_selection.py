import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.gaussian_process.kernels import (
    Matern,
    RationalQuadratic,
    ConstantKernel,
    RBF,
)


class ModelSelection:
    """
    Wrapper class on sklearn model / features selections
    """

    def __init__(self, X_train, y_train, X_dev, y_dev, params_dict=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_dev = X_dev
        self.y_dev = y_dev

        if params_dict:
            self.params_dict = params_dict
        else:
            self.params_dict = {
                "kernel": [
                    ConstantKernel() * RBF(),
                    ConstantKernel() * Matern(),
                    ConstantKernel() * RationalQuadratic(),
                ]
            }

    def model_tuning(self, model):
        """
        Given a model and a dictionary of parameter to be tested,
        perform a grid search
        :param model: the model to be tuned
        :return: result of the grid search on the different folds and
        the best estimator
        """
        X = np.vstack([self.X_train, self.X_dev])
        y = np.concatenate([self.y_train, self.y_dev])
        test_fold = [-1 if i <= len(self.X_train) else 0 for i in range(len(X))]
        model_grid_search = GridSearchCV(
            model,
            self.params_dict,
            cv=PredefinedSplit(test_fold=test_fold),
            verbose=2,
            n_jobs=5,
        )
        model_grid_search.fit(X, y)
        return model_grid_search.best_estimator_
