#!/usr/bin/env python  
# -*- encoding:utf-8 -*-  
# author: zy
# time: 2021/2/6 10:27 
# file: XGBoost.py 

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


#网格搜索寻求最优参数
def cv(unif_train_x, train_y):
    # 分类器使用 xgboost
    clf1 = xgb.XGBRegressor()

    # 设定网格搜索的xgboost参数搜索范围，搜索XGBoost的主要6个参数
    param_dist = {
        # 'n_estimators': range(80, 200, 4),
        'n_estimators': range(5, 80, 1),
        'max_depth': range(2, 15, 1),
        'learning_rate': np.linspace(0.01, 2, 20),
        # 'subsample': np.linspace(0.7, 0.9, 20),
        # 'colsample_bytree': np.linspace(0.5, 0.98, 10),
        # 'min_child_weight': range(1, 9, 1)
    }

    # GridSearchCV参数说明，clf1设置训练的学习器
    # param_dist字典类型，放入参数搜索范围
    # scoring = 'neg_log_loss'，精度评价方式设定为“neg_log_loss“
    # n_iter=300，数值越大，获得的参数精度越大，但是搜索时间越长
    # n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
    grid = GridSearchCV(clf1, param_dist, cv=5, n_jobs=2)

    # 在训练集上训练
    grid.fit(unif_train_x, train_y)
    # 返回最优的训练器
    best_estimator = grid.best_estimator_
    best_params = grid.best_params_
    # 输出最优训练器的精度
    print("最优训练器的精度\n", best_estimator)

    return best_estimator, best_params

def train_model(train_x, train_y, test_x, test_y, other_params):
    grid_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                 importance_type='gain', interaction_constraints='',
                 learning_rate=0.9526315789473684, max_delta_step=0, max_depth=3,
                 min_child_weight=1, missing=None, monotone_constraints='()',
                 n_estimators=14, n_jobs=0, num_parallel_tree=1,
                 objective='reg:squarederror', random_state=0, reg_alpha=0,
                 reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
                 validate_parameters=1, verbosity=None)
    other_params = other_params
    # grid_model = xgb.XGBRegressor(**other_params)
    model = grid_model.fit(train_x, train_y)
    pre = model.predict(test_x)

    plot_importance(model, max_num_features=11, height=0.5, importance_type='weight')
    plt.show()
    print("R方值：", model.score(test_x, test_y))
    print("相对误差率为 ", ((abs(pre - test_y) / test_y) * 100))

    plt.plot(range(0, pre.shape[0]), pre, 'r', lw=4, marker='s', mec='r', mfc='r', markersize=8)
    plt.plot(range(0, test_y.shape[0]), test_y, 'b', lw=4, marker='o', mec='b', mfc='b', markersize=8)
    plt.yticks(np.arange(1100, 3100, step=100))
    plt.legend(["pre", "test_Y"])
    plt.show()

#使用函数随机划分数据
def data():
    # 导入训练数据2
    c = pd.read_excel('your data')
    x = c.iloc[:, :-1]
    y = c.iloc[:, -1]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=22)  # , random_state=543
    col_name = list(train_x.columns)

    # 数据标准化
    Scaler = MinMaxScaler().fit(x)  # 根据data_train生成离差标准化规则
    Scaler_train_x = Scaler.transform(train_x)  # 将规则应用于训练集，即对训练集进行离差标准化处理
    Scaler_test_x = Scaler.transform(test_x)  # 将规则应用于测试集
    unif_train_x = pd.DataFrame(Scaler_train_x, columns=col_name)
    unif_test_x = pd.DataFrame(Scaler_test_x, columns=col_name)

    return unif_train_x, train_y, unif_test_x, test_y

if __name__ == '__main__':

    unif_train_x, train_y, unif_test_x, test_y = data()

    best_params = []
    # 网格搜索最优参数
    # best_estimator, best_params = cv(unif_train_x, train_y)

    # 模型训练
    train_model(unif_train_x, train_y, unif_test_x, test_y, best_params)
