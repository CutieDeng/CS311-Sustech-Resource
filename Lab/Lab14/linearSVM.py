import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class SVM(object):
    def __init__(self, x, y, epochs=200, learning_rate=0.01):
        self.x = np.c_[np.ones((x.shape[0])), x]
        self.y = y
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.w = np.random.uniform(size=np.shape(self.x)[1],)
        print(self.x)
        print(self.w)

    def get_loss(self, x, y):
        loss = max(0, 1 - y * np.dot(x, self.w))
        return loss
    
    def cal_sgd(self, x, y, w):
        if y * np.dot(x, w) < 1:
            w = w - self.learning_rate * (-y * x)
        else:
            w = w
        
        return w

    def train(self):
        for epoch in range(self.epochs):
            randomize = np.arange(len(self.x))
            np.random.shuffle(randomize)
            x = self.x[randomize]
            y = self.y[randomize]
            loss = 0            
            for xi, yi in zip(x, y):
                loss += self.get_loss(xi, yi)
                self.w = self.cal_sgd(xi, yi, self.w)          
            print('epoch: {0} loss: {1}'.format(epoch, loss))

    def predict(self, x):
        x_test = np.c_[np.ones((x.shape[0])), x]
        return np.sign(np.dot(x_test, self.w))


def main():
    #Training Data
    x = np.array([[1,2],[3,2],[3,4],[7,2],[10,1],[7,3],[11,4],[13,3]])
    y = np.array([1,1,1,-1,-1,-1,-1,-1])

    #Testing Data
    x_test=np.array([[1,2],[3,2],[3,4],[7,2],[10,1],[7,3],[11,4],[13,3]])

    mysvm = SVM(x, y)
    mysvm.train()
    pre_y = mysvm.predict(x_test)
    print(pre_y)
    print(np.sum((y-pre_y) == 0)/pre_y.size)

if __name__ == '__main__':
    main()
