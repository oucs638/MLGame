import pickle
import numpy as np
import os
from os import path
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


def transformCommand(command):
    if 'RIGHT' in str(command):
        return 2
    elif 'LEFT' in str(command):
        return 1
    else:
        return 0
    pass


def get_vector(now, pre):
    return now - pre


def get_ArkanoidData(filename):
    Frames = []
    i = 0
    Balls = []
    Vectors = []
    Commands = []
    s = []
    PlatformPos = []
    log = pickle.load((open(filename, 'rb')))
    for sceneInfo in log:
        if i == 0:
            Vectors.append([0, 0])
        else:
            Vectors.append([get_vector(sceneInfo.ball[0], prex),
                            get_vector(sceneInfo.ball[1], prey)])
        i += 1
        Frames.append(sceneInfo.frame)
        Balls.append([sceneInfo.ball[0], sceneInfo.ball[1]])
        prex = sceneInfo.ball[0]
        prey = sceneInfo.ball[1]
        PlatformPos.append(sceneInfo.platform)
        Commands.append(transformCommand(sceneInfo.command))
    commands_ary = np.array([Commands])
    commands_ary = commands_ary.reshape((len(Commands), 1))
    frame_ary = np.array(Frames)
    frame_ary = frame_ary.reshape((len(Frames), 1))
    # data = np.hstack((frame_ary, Vectors, PlatformPos, commands_ary))
    data = np.hstack((frame_ary, Balls, PlatformPos, commands_ary))
    return data


if __name__ == '__main__':
    # filename = path.join(path.dirname(_file_), 'arkanoid_dataset.pickle')
    logfile = './games/arkanoid/log/'
    for file in os.listdir(logfile):
        filename = path.join(logfile, file)
        # print(filename)
        data = get_ArkanoidData(filename)
        data = data[1::]
        Balls = data[:, 1:3]
        Balls_next = np.array(Balls[1:])
        vectors = Balls_next - Balls[:-1]
        direction = []
        for i in range(len(data)-1):
            if(vectors[i, 0] > 0 and vectors[i, 1] > 0):
                direction.append(0)  # 向右上為0
            elif(vectors[i, 0] > 0 and vectors[i, 1] < 0):
                direction.append(1)  # 向右下為1
            elif(vectors[i, 0] < 0 and vectors[i, 1] > 0):
                direction.append(2)  # 向左上為2
            elif(vectors[i, 0] < 0 and vectors[i, 1] < 0):
                direction.append(3)  # 向左下為3

        direction = np.array(direction)
        direction = direction.reshape((len(direction), 1))
        data = np.hstack((data[1:, :], direction))
        ball_x = data[:, 1]  # data[:,1]
        data[:, 1] = vectors[:, 0]
        ball_y = data[:, 2]  # data[:,2]
        data[:, 2] = vectors[:, 1]
        mask = [1, 2, 3, 6]
        X = data[:, mask]
        Y = data[:, -2]
        Direct = data[:, -1]
    # print(Y)
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2)   # 分割資料成測試和訓練
    # platform_predict_clf = KMeans(n_clusters=3, random_state=0).fit(
    # x_train, y_train)    # 模型的訓練
    knn = KNeighborsClassifier().fit(x_train, y_train)

    # y_predict = platform_predict_clf.predict(x_test)    # classifier
    y_predict = knn.predict(x_test)
    print(y_predict)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    print("Accuracy(正確率) ={:8.3f}%".format(accuracy*100))

    ax = plt.subplot(111, projection='3d')
    ax.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1],
               X[Y == 0][:, 3], c='#FF0000', alpha=1)
    ax.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1],
               X[Y == 1][:, 3], c='#2828FF', alpha=1)
    ax.scatter(X[Y == 2][:, 0], X[Y == 2][:, 1],
               X[Y == 2][:, 3], c='#007500', alpha=1)

    plt.title("KMeans Prediction")
    ax.set_xlabel('v_x')
    ax.set_ylabel('v_y')
    ax.set_zlabel('Direction')

    plt.show()

    with open('./games/arkanoid/ml/save/clf_KMeans_BallAndDirection.pickle', 'wb') as f:
        # pickle.dump(platform_predict_clf, f)
        pickle.dump(knn, f)
