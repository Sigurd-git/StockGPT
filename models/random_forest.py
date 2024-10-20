import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA


def random_forest_reg(
    train_data, test_data, feature_names, output_variable, n_components=20
):
    """
    Train a RandomForestRegressor model and predict the closing prices.

    Parameters:
    - train_data: DataFrame containing the training data.
    - test_data: DataFrame containing the test data.
    - feature_names: List of feature names to be used for training.
    - output_variable: The target variable for prediction.
    - n_components: Number of components for PCA, if applicable.

    Returns:
    - predictions: Predicted values for the test data.
    """
    # Initialize model and variables
    rf_model = RandomForestRegressor()
    train_output = train_data[output_variable]

    # Apply PCA if necessary
    if len(feature_names) > n_components:
        pca_model = PCA(n_components=n_components)
        train_data = pca_model.fit_transform(train_data[feature_names])
        test_data = pca_model.transform(test_data[feature_names])
    else:
        train_data = train_data[feature_names]
        test_data = test_data[feature_names]

    # Fit model and make predictions
    rf_model.fit(train_data, train_output)
    predictions = rf_model.predict(test_data)

    return predictions
