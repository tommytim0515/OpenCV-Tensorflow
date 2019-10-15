#Name: TIAN Xiangan
#ITSC: xtianae
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

lowerThres = np.array([65, 50, 120])
upperThres = np.array([95, 210, 255])

def generateImg(loewrBound, upperBound, imgNum):
    cap = cv2.VideoCapture('17_43_50Uncompressed-0000.avi')
    counter = 0
    global bitXor

    while counter < loewrBound:
        _, frame = cap.read()
        if frame is None:
            break
        counter += 1
    
    while counter <= upperBound:
        _, frame = cap.read()
        if frame is None:
            break
        counter += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerThres, upperThres)
        mask = cv2.medianBlur(mask, 5)
        res = cv2.bitwise_or(frame, frame, mask = mask)
        kernel = np.ones((7, 7), np.float32) / 49
        smoothed = cv2.filter2D(res, -1, kernel)

        if counter == loewrBound  + 1:
            bitXor = smoothed
        else:
            bitXor = cv2.bitwise_xor(bitXor, smoothed)

        print("Processed No." + str(counter - 1) + " Image.")
        
    cv2.imwrite('trajectory' + str(imgNum) + '.jpg', bitXor)
    print('Finished Generation Trajectory Image(%d)' % imgNum)

def interpolationTrajectory(index):
    img = cv2.imread('trajectory%d.jpg' % index)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB_FULL)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(img,200,250,apertureSize = 5) 

    x_set = np.array([], dtype = float)
    y_set = np.array([], dtype = float)
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i, j] > 250: # and j < 1000 and j > 200:
                # print('[' + str(i) + ', ' + str(j) + ', ' + str(edges[i, j]) + ']')
                x_set = np.append(x_set, (j / 1280))
                y_set = np.append(y_set, (i / 1024))

    # plt.plot(x_set, y_set, 'o')
    # plt.show()
    # return
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    a = tf.Variable(np.random.randn(), name = 'para1')
    b = tf.Variable(np.random.randn(), name = 'para2')
    c = tf.Variable(np.random.randn(), name = 'para3')
    pred = a*X*X + b*X + c
    cost = tf.reduce_sum((pred - Y) ** 2) / (2 * len(x_set))

    learning_rate = 0.5#0.3
    epochs = 1000#200

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sesh:
        sesh.run(init)
        print("Linear Regression Img" + str(index))

        for epoch in range(epochs):
            for x, y in zip(x_set, y_set):
                sesh.run(optimizer, feed_dict = {X: x, Y: y})
            if not epoch % 10:
                Cost = sesh.run(cost, feed_dict = {X: x_set, Y: y_set})
                P1 = sesh.run(a)
                P2 = sesh.run(b)
                P3 = sesh.run(c)
                print("Epoch: %3d, Cost: %.4f, a: %.4f, b: %.4f, c: %.4f" % (epoch, Cost, P1, P2, P3))
                if Cost < 0.0007:
                    learning_rate = 0.1
                elif Cost < 0.0004:
                    learning_rate = 0.04
                elif Cost < 0.0002:
                    break

        Cost = sesh.run(cost, feed_dict = {X: x_set, Y: y_set})
        para1 = sesh.run(a)
        para2 = sesh.run(b)
        para3 = sesh.run(c)
        print("Finalized Cost: %.4f A: %.4f B: %.4f C: %.4f" % (Cost, para1, para2, para3))


        plt.plot(x_set, y_set, 'o')
        plt.plot(x_set, para1*x_set*x_set + para2*x_set + para3)
        plt.show()

        # x1 = 0
        # y1 = int(1024*para2)
        # x2 = 1280
        # y2 = int(1024*(para1 + para2))
        
        # result = cv2.imread('trajectory%d.jpg' % index)
        # cv2.line(result, (x1, y1), (x2, y2), (240, 100, 100), 2)
        # cv2.imwrite('interpolation%d.jpg' % index, result)
    print("Finished Linear Regression Img" + str(index))


def skLearnLR(index):
    img = cv2.imread('trajectory%d.jpg' % index)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB_FULL)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(img,200,250,apertureSize = 5) 

    x_set = np.array([], dtype = float)
    y_set = np.array([], dtype = float)
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i, j] > 240:
                x_set = np.append(x_set, j)
                y_set = np.append(y_set, i)
    # x_mean = np.mean(x_set)
    y_mean = np.mean(y_set)
    i = 0
    while i < len(x_set):
        if (abs(y_set[i] - y_mean) / y_mean)  > 0.3:
            x_set = np.delete(x_set, i)
            y_set = np.delete(y_set, i)
        else:
            i += 1
    x_set = x_set.reshape(-1, 1)
    y_set = y_set.reshape(-1, 1)
    regressor = LinearRegression().fit(x_set, y_set)
    x1 = 0
    y1 = int(regressor.intercept_)
    x2 = 1280
    y2 = int(1280*regressor.coef_ + regressor.intercept_)    
    result = cv2.imread('trajectory%d.jpg' % index)
    cv2.line(result, (x1, y1), (x2, y2), (240, 100, 100), 5)
    cv2.imwrite('interpolation%d.jpg' % index, result)
    # plt.plot(x_set, y_set, 'o')
    # plt.plot(x_set, regressor.coef_*x_set + regressor.intercept_)
    # plt.show()
    
generateImg(39, 41, 1)
generateImg(43, 45, 2)
generateImg(47, 49, 3)
generateImg(51, 53, 4)
generateImg(54, 56, 5)
generateImg(57, 59, 6)
generateImg(61, 63, 7)
generateImg(72, 74, 8)
generateImg(122, 124, 9)
generateImg(142, 145, 10)
generateImg(179, 182, 11)
generateImg(209, 213, 12)
generateImg(252, 254, 13)
generateImg(280, 281, 14)
generateImg(303, 305, 15)
generateImg(315, 317, 16)

# for i in range(1, 17):
# skLearnLR(15)
    # print('Finished No. ' + str(i))
    # interpolationTrajectory(i)
# interpolationTrajectory(15)