import numpy as np
from error_evaluate import evaluate_error
from sklearn import preprocessing
from sklearn import linear_model
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
# lr = linear_model.LinearRegression().fit(x_train, y_train)
# lr = linear_model.Ridge(alpha=0.01).fit(x_train, y_train)
# lr = linear_model.Lasso(alpha=0.0001, max_iter=50000).fit(x_train, y_train)
# y_train_predict = lr.predict(x_train)
# y_test_predict = lr.predict(x_test)


"""保存"""
# joblib.dump(lr, 'result/lr_model.pkl')
# np.save('result/y_train_predict_lr.npy', y_train_predict)
# np.save('result/y_test_predict_lr.npy', y_test_predict)
# np.save('result/y_train_predict_ridge.npy', y_train_predict)
# np.save('result/y_test_predict_ridge.npy', y_test_predict)
# np.save('result/y_train_predict_lasso.npy', y_train_predict)
# np.save('result/y_test_predict_lasso.npy', y_test_predict)

"""误差"""
# lr = joblib.load('result/lr_model.pkl')
y_train_predict = np.load('result/y_train_predict_lasso.npy')
y_test_predict = np.load('result/y_test_predict_lasso.npy')
error_train = evaluate_error(y_train, y_train_predict)
error_test = evaluate_error(y_test, y_test_predict)
print(f'error of training data: {error_train}')
print(f'error of test data: {error_test}')
