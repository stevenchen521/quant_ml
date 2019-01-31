from sklearn import preprocessing
import numpy as np
import math

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
# X_scaled = preprocessing.scale(X_train)
mm = preprocessing.StandardScaler()
mm_data = mm.fit_transform(X_train)
mean = mm.mean_[-1]
std = math.sqrt(mm.var_[-1])
trans = mm_data[:, 2]
func = np.vectorize(lambda x: x * std + mean)
func(trans)
# predict = [9,2,10]
# origin_data = mm.inverse_transform(mm_data[:, 2])

