from my_ml_lib.nn import *
from my_ml_lib.nn.modules import *

class MySoftMax(Module):
    def __init__(self , n_features , n_classes):
        super().__init__()

        self.network = Sequential(Linear(n_features , n_classes))

    def __call__(self , X):
        return self.network(X)
    
    def __repr__(self):
        # Delegate the representation to the internal Sequential network
        return f"{self.__class__.__name__}(\n{repr(self.network)}\n)"
    

def initialize_best_model():
    model = MySoftMax(784 , 10)
    return model.network
        


