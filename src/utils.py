# The feature selector class and processor function to handle transformation of new data

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_type='numerical'):
        self.feature_type = feature_type
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.feature_type == 'numerical':
            return X.select_dtypes(exclude=['object'])
        elif self.feature_type == 'categorical':
            return X.select_dtypes(include=['object'])

def create_preprocessor(X_train):
    # Getting categorical and numerical features
    def get_feature_types(data):
        categorical_features = data.select_dtypes(include=['object']).columns
        numerical_features = data.select_dtypes(exclude=['object']).columns
        return categorical_features, numerical_features

    # Getting features for the training data
    categorical_features, numerical_features = get_feature_types(X_train)

    # Creating a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('selector', FeatureSelector(feature_type='numerical')),
                ('scaler', MinMaxScaler())
            ]), numerical_features),
            ('cat', Pipeline(steps=[
                ('selector', FeatureSelector(feature_type='categorical')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])
    
    return preprocessor

def create_pipeline(X_train, y_train, model):
    preprocessor = create_preprocessor(X_train)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)

    return pipeline