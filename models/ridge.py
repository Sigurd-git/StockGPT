import torch
import numpy as np
from himalaya.ridge import RidgeCV


def ridge_reg(train_data, test_data, feature_names, output_variable):
    # Prepare data
    X_train = torch.tensor(train_data[feature_names].values, dtype=torch.float32)
    y_train = torch.tensor(
        train_data[output_variable].values, dtype=torch.float32
    ).reshape(-1, 1)
    X_test = torch.tensor(test_data[feature_names].values, dtype=torch.float32)

    # Standardize data
    mean_train = X_train.mean(dim=0)
    std_train = X_train.std(dim=0)
    X_train = (X_train - mean_train) / std_train
    X_test = (X_test - mean_train) / std_train

    ridge = RidgeCV(alphas=np.logspace(-50, 50, 100, base=2))
    ridge.fit(X_train, y_train)
    predictions = ridge.predict(X_test)
    return predictions
