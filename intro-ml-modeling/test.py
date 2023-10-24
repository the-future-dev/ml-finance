import numpy as np
from code.model import build_and_train_model

num_features = 5  # or replace 5 with X_train.shape[1] if X_train is defined somewhere

test_X = np.random.rand(100, num_features)
test_y = np.random.rand(100, 1)

build_and_train_model(test_X, test_y, 0.01)