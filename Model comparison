import pandas as pd
from sklearn import model_selection
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor#决策树回归
from sklearn.ensemble import RandomForestRegressor#随机森林回归
from sklearn.ensemble import GradientBoostingRegressor#梯度上升回归
from sklearn.neural_network import MLPRegressor #BP神经网络
from sklearn.svm import SVR #支持向量机回归
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import r2_score # R square
from sklearn.preprocessing import MinMaxScaler


def model_BP(train_x, train_y, test_x, test_y):
    model = MLPRegressor(hidden_layer_sizes=(11,), random_state=10,learning_rate_init=0.1)          # BP神经网络回归模型
    model.fit(train_x, train_y)
    pred_y_test = model.predict(test_x)
    print("BP神经网络_R方值：", model.score(test_x, test_y))
    print(abs((pred_y_test - test_y)/test_y*100))
    return model.score(test_x, test_y), pred_y_test


def model_SpportVctorRgression(train_x, train_y, test_x, test_y):
    model = SVR(kernel='linear',  cache_size=1000)   #线性核函数初始化的SVR
    model.fit(train_x,train_y)
    pred_y_test = model.predict(test_x)
    print("SpportVctorRgression_R方值：", model.score(test_x, test_y))
    print(abs((pred_y_test - test_y)/test_y*100))
    return model.score(test_x, test_y), pred_y_test

def model_DecisionTreeRegressor(train_x, train_y, test_x, test_y):
    model = DecisionTreeRegressor()
    model.fit(train_x, train_y)
    pred_y_test = model.predict(test_x)
    print("DecisionTreeRegressor_R方值：", model.score(test_x, test_y))
    print(test_y - pred_y_test)
    return model.score(test_x, test_y),  pred_y_test

def model_RandomForestRegressor(train_x, train_y, test_x, test_y):
    model = RandomForestRegressor()
    model.fit(train_x, train_y)
    pred_y_test = model.predict(test_x)
    print("RandomForestRegressor_R方值：", model.score(test_x, test_y))
    print(test_y - pred_y_test)
    return model.score(test_x, test_y), pred_y_test

def model_GradientBoostingRegressor(train_x, train_y, test_x, test_y):
    model = GradientBoostingRegressor()
    model.fit(train_x, train_y)
    pred_y_test = model.predict(test_x)
    print("GradientBoostingRegressor_R方值：", model.score(test_x, test_y))
    print(test_y - pred_y_test)
    return model.score(test_x, test_y),  pred_y_test

def model_LinearRegression(train_x, train_y, test_x, test_y):
    model = LinearRegression()
    model.fit(train_x, train_y)
    pred_y_test = model.predict(test_x)
    print("LinearRegression_R方值：", model.score(test_x, test_y))
    print(test_y - pred_y_test)
    return model.score(test_x, test_y),  pred_y_test

if __name__ == '__main__':
    train = pd.read_excel(r'C:\Users\zy\Desktop\train.xlsx')
    test = pd.read_excel(r'C:\Users\zy\Desktop\test.xlsx')
    train_x = train.iloc[:, :-1]
    train_y = train.iloc[:, -1]
    test_x = test.iloc[:, :-1]
    test_y = test.iloc[:, -1]

    # 数据标准化
    Scaler = MinMaxScaler().fit(train.iloc[:, :-1])  # 根据data_train生成离差标准化规则
    unif_train_x = Scaler.transform(train_x)  # 将规则应用于训练集，即对训练集进行离差标准化处理
    unif_test_x = Scaler.transform(test_x)  # 将规则应用于测试集

    # train_x, test_x, train_y, test_y = model_selection.train_test_split(x, y, test_size=0.2, random_state=22)#random_state=22

    print(test_x)
    print(test_y)
    models = [
        model_DecisionTreeRegressor(unif_train_x, train_y, unif_test_x, test_y),
        model_RandomForestRegressor(unif_train_x, train_y, unif_test_x, test_y),
        model_GradientBoostingRegressor(unif_train_x, train_y, unif_test_x, test_y),
        model_LinearRegression(unif_train_x, train_y, unif_test_x, test_y),
        model_SpportVctorRgression(train_x, train_y, test_x, test_y),
        model_BP(train_x, train_y, test_x, test_y)
    ]
    Rlist = []
    prelist = []
    for model in models:
        R, pre = model
        Rlist.append(R)
        prelist.append(pre)


    print(list(test_y))
    print(pd.DataFrame(Rlist))
    print(pd.DataFrame(prelist))
    rr = pd.DataFrame(Rlist)
    pp = pd.DataFrame(prelist)
    prt = pd.merge(pp, rr, how='outer')
    print(prt)

