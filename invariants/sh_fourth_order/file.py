
import nrrd
import numpy as np

def load_transform(file_path, l):

    train_data, options = nrrd.read(file_path)
    x = train_data.transpose()


    # read all datasets
X_train_1, y_train_1 = load_transform('one-train-odfs.nrrd', 0)
X_train_2, y_train_2 = load_transform('two-train-odfs.nrrd', 1)
X_train_3, y_train_3 = load_transform('three-train-odfs.nrrd', 2)
X_test_1, y_test_1 = load_transform('one-test-odfs.nrrd', 0)
X_test_2, y_test_2 = load_transform('two-test-odfs.nrrd', 1)
X_test_3, y_test_3 = load_transform('three-test-odfs.nrrd', 2)

# merge datasets
X_train_temp = np.append(X_train_1, X_train_2, axis=0)
X_train = np.append(X_train_temp, X_train_3, axis=0)

y_train_temp = np.append(y_train_1, y_train_2, axis=0)
y_train = np.append(y_train_temp, y_train_3, axis=0)

X_test_temp = np.append(X_test_1, X_test_2, axis=0)
X_test = np.append(X_test_temp, X_test_3, axis=0)

y_test_temp = np.append(y_test_1, y_test_2, axis=0)
y_test = np.append(y_test_temp, y_test_3, axis=0)