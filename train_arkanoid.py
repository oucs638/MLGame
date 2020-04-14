import pickle
import numpy as np
import os
from os import path
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
# from mpl_toolkits.mplot3d import Axes3D


def transformCommand(command):
    if 'RIGHT' in str(command):
        return 1
    elif 'LEFT' in str(command):
        return -1
    else:
        return 0
    pass


def transformStatus(status):
    if 'GAME_OVER' in str(status) or 'GAME_PASS' in str(status):
        return 0
    else:
        return 1


def get_ArkanoidData(filename):
    Status = []
    Balls = []
    PlatformPos = []
    Commands = []
    log = pickle.load((open(filename, 'rb')))
    for sceneInfo in log:
        Status.append(transformStatus(sceneInfo.status))
        Balls.append([sceneInfo.ball[0], sceneInfo.ball[1]])
        PlatformPos.append(sceneInfo.platform)
        Commands.append(transformCommand(sceneInfo.command))
    status_ary = np.array([Status])
    status_ary = status_ary.reshape((len(Status), 1))
    commands_ary = np.array([Commands])
    commands_ary = commands_ary.reshape((len(Commands), 1))
    log_data = np.hstack((status_ary, Balls, PlatformPos, commands_ary))
    return log_data


def get_vector_pos(bal_now, bal_pre, plt_pos):
    bal_xsp = bal_now[0] - bal_pre[0]
    bal_ysp = bal_now[1] - bal_pre[1]
    bal_siz, hit_max = 5, 195
    res_pos = bal_now[0]
    if bal_ysp > 0:
        tp_time = (plt_pos[1] - bal_siz - bal_now[1]) // bal_ysp
        res_pos = abs(bal_now[0] + bal_xsp * tp_time)
        if res_pos > hit_max:
            if (res_pos // hit_max) % 2 == 0:
                res_pos -= (res_pos // hit_max) * hit_max
            else:
                res_pos -= (res_pos // hit_max) * hit_max
                res_pos = hit_max - res_pos
    else:
        res_pos = 100
    return bal_now, res_pos


def preprocess_data(log_data):
    bal_pre = [93, 395]
    plt_xsz, bal_siz = 40, 5
    data_x, data_y = [], []
    for frm in log_data:
        if frm[0] == 0:
            data_x.append(int(0))
            data_y.append(int(0))
            bal_pre = [frm[1], frm[2]]
        else:
            bal_pre, res_pos = get_vector_pos(
                [frm[1], frm[2]], bal_pre, [frm[3], frm[4]])
            if res_pos < (frm[3] + bal_siz):
                data_x.append(int(-1))
                data_y.append(int(-1))
            elif res_pos > (frm[3] + plt_xsz - bal_siz * 2):
                data_x.append(int(1))
                data_y.append(int(1))
            else:
                data_x.append(int(0))
                data_y.append(int(0))
    return data_x, data_y


if __name__ == '__main__':
    dataX, Y = [], []
    logfile = './games/arkanoid/log/'
    for file in os.listdir(logfile):
        filename = path.join(logfile, file)
        data = get_ArkanoidData(filename)
        data_x, data_y = preprocess_data(data)
        dataX += data_x
        Y += data_y
    X = []
    for i in range(len(dataX)):
        X.append([dataX[i], dataX[i]])

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.5, random_state=42)
    clf = ExtraTreesClassifier(n_estimators=15, max_depth=None,
                               min_samples_split=3, random_state=0).fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    print("Accuracy(正確率) ={:8.3f}%".format(accuracy*100))

    # # ax = plt.subplot(111, projection='3d')
    # # ax.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1],
    # #            X[Y == 0][:, 3], c='#FF0000', alpha=1)
    # # ax.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1],
    # #            X[Y == 1][:, 3], c='#2828FF', alpha=1)
    # # ax.scatter(X[Y == 2][:, 0], X[Y == 2][:, 1],
    # #            X[Y == 2][:, 3], c='#007500', alpha=1)
    # # plt.title("KNN Prediction")
    # # ax.set_xlabel('v_x')
    # # ax.set_ylabel('v_y')
    # # ax.set_zlabel('Direction')
    # # plt.show()

    with open('./games/arkanoid/ml/save/clf_RANF_BallAndDirection.pickle', 'wb') as f:
        pickle.dump(clf, f)
