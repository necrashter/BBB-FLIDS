import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset
from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler # used for feature scaling
from sklearn.compose import ColumnTransformer 

class DataPreprocessor:
    """
    Preprocess data, apply one-hot encoding to categorical data etc.
    """
    def __init__(self, test_size=0.2):
        self.test_size = test_size

    def fit_load(self, filename):
        """
        1. Load the given CSV from filename.
        2. Create and fit the encoders/transformers to it.
        """
        dataset = pd.read_csv(filename)
        # Splitting the attributes into independent and dependent attributes
        X = dataset.iloc[:, :-1].values # attributes to determine dependent variable / Class
        Y = dataset.iloc[:, -1].values # dependent variable / Class

        # Encode categorical data
        self.categorical_features = [i for i, e in enumerate(X[0]) if isinstance(e, str)]
        self.column_transformer = ColumnTransformer([("categorical", OneHotEncoder(), self.categorical_features)], remainder="passthrough")
        X = self.column_transformer.fit_transform(X)

        # Handle categorical class data
        self.y_transformer = LabelEncoder()
        Y = self.y_transformer.fit_transform(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=0)

        # NOTE: Scaling is not done here
        # In federated learning, participants don't know the data of other users
        self.x_scaler = StandardScaler(with_mean=False, with_std=False)
        X_train = self.x_scaler.fit_transform(X_train)
        X_test = self.x_scaler.transform(X_test)

        return X_train, X_test, Y_train, Y_test

