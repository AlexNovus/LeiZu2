import numpy as np
from error_evaluate import evaluate_error
from sklearn import preprocessing
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import joblib

"""读取数据"""
x_train = np.load('data/x_train.npy')
x_test = np.load('data/x_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')
index_test = np.load('data/y_test.npy')

"""数据预处理"""
scaler = preprocessing.MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

"""训练"""
# svr = MultiOutputRegressor(SVR(kernel='rbf', C=2), n_jobs=30).fit(x_train, y_train)
# svr = MultiOutputRegressor(SVR(kernel='linear', C=9), n_jobs=30).fit(x_train, y_train)
# y_train_predict = svr.predict(x_train)
# y_test_predict = svr.predict(x_test)


"""保存"""
# joblib.dump(svr, 'svr_model.pkl')
# np.save('y_train_predict_svr_linear.npy', y_train_predict)
# np.save('y_test_predict_svr_linear.npy', y_test_predict)

"""误差"""
# lr = joblib.load('svr_model.pkl')
y_train_predict = np.load('result/y_train_predict_svr_linear.npy')
y_test_predict = np.load('result/y_test_predict_svr_linear.npy')
error_train = evaluate_error(y_train, y_train_predict)
error_test = evaluate_error(y_test, y_test_predict)
print(f'error of training data: {error_train}')
print(f'error of test data: {error_test}')
