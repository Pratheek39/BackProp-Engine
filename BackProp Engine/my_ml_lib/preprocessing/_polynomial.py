import numpy as np
# May need itertools.combinations_with_replacement
from itertools import combinations_with_replacement
import math

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias
        # Add any other attributes needed to store fitted info if necessary

    def fit(self, X, y=None):
        # Usually fit doesn't do much here unless you need input dimensions
        self.n_input_features_ = X.shape[1]
        return self

    def transform(self, X):
        
        # TODO: Generate polynomial features up to self.degree
        # Remember to handle include_bias
        #print("PolynomialFeatures: Transform method needs implementation.")
        # Example for degree 2, include_bias=True, X = [a, b]
        # Output should be [1, a, b, a^2, ab, b^2]
        # Consider using sklearn's implementation as a reference for a robust solution
        X_transformed = []
        for x in X:
            x_feature = [1.0]
            for i in range(self.degree):
                x_ = combinations_with_replacement(x , i+1 )
                for tup in x_:
                    val = math.prod(tup)
                    x_feature.append(val)
            X_transformed.append(x_feature)


        return np.array(X_transformed) # Placeholder

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    

