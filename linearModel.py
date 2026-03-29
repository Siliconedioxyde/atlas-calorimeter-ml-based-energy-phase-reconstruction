import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Load all training data

X_train_list = []
y_train_list = []

train_files = sorted( glob.glob("data/train/*.pt"))

for filename in train_files:
    data = torch.load(filename)

    X_lg = data['X'][:, 1, :].numpy()
    y_lg = data['y'][:, 1].numpy()

    X_train_list.append(X_lg)
    y_train_list.append(y_lg)

X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)

print("Training data shape:", X_train.shape)

#Training the model using Linear Regression

model = LinearRegression()
model.fit(X_train, y_train)

weights = model.coef_

print("\nLearned weights:")
print(weights)

#Again load all test files

test_files = sorted( glob.glob("data/test/*.pt"))

X_test_list = []
y_test_list = []

for file in test_files:
    data = torch.load(file)

    X_lg = data['X'][:, 1, :].numpy()
    y_lg = data['y'][:, 1].numpy()

    X_test_list.append(X_lg)
    y_test_list.append(y_lg)

X_test = np.vstack(X_test_list)
y_test = np.concatenate(y_test_list)

print("Test data shape:", X_test.shape)

# Evaluate on the test file

y_pred = model.predict(X_test)


relative_residual = (y_pred - y_test) / y_test

mean_residual = np.mean(relative_residual)
rms_residual = np.std(relative_residual)

print("\n--- TEST PERFORMANCE ---")
print("Mean residual:", mean_residual)
print("RMS residual:", rms_residual)

# Relative Residual Histogram

plt.figure()
plt.hist(relative_residual, bins=400000,log=0)
plt.xlabel("(reco - true) / true")
plt.ylabel("Counts")
plt.title("Relative Residual Distribution")
plt.show()

# Relative Residual vs True Energy

plt.figure()
plt.scatter(y_test, relative_residual, s=3)
plt.xlabel("True Energy")
plt.ylabel("(reco - true) / true")
plt.title("Residual vs True Energy")
plt.show()

# Predicted Energy vs True Energy

plt.figure()
plt.scatter(y_test, y_pred, s=3, c='red')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
plt.xlabel("True Energy")
plt.ylabel("Predicted Energy")
plt.title("Predicted vs True Energy")
plt.show()

