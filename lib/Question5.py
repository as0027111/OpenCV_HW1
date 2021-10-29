import tensorflow as tf
import keras
import os 
import random
import numpy as np
import keras
import tensorflow as tf
import pickle
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers
import cv2


class Q5():
    def __init__(self) -> None:
        self.batch_size = 32 
        self.epochs = 50
        self.num_classes = 10
        self.lr = 0.0001
        self.opt = tf.keras.optimizers.Adam(lr=self.lr)
        filename = ['data_batch_1','data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        self.label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.folder = './Dataset_CvDl_Hw1/cifar-10-python/'
        '''
        x1, y1 = self.load_data(self.folder+filename[0])
        x2, y2 = self.load_data(self.folder+filename[1])
        x3, y3 = self.load_data(self.folder+filename[2])
        x4, y4 = self.load_data(self.folder+filename[3])
        x5, y5 = self.load_data(self.folder+filename[4])
        self.x_train = np.concatenate((x1, x2, x3, x4, x5))
        
        self.y = np.concatenate((y1, y2, y3, y4, y5))
        y_train = keras.utils.to_categorical(self.y, 10)
        '''
        #print(x, y)
        self.x_test, self.y_test_show = self.load_data(self.folder+'test_batch')
        self.y_test = keras.utils.to_categorical(self.y_test_show, 10)
        self.model = tf.keras.models.load_model('model.h5', compile=False)

        #print(np.size(x), np.size(y))
        # print(self.x_train.shape[1:]) #there are 50000 32*32 pictures with RGB channel 

    def load_data(self, file):
        DataDecorder={}
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            for itemname,dataset in dict.items():
                DataDecorder[itemname.decode('utf8')]=dataset
        dataset = DataDecorder['data']
        label = DataDecorder['labels']
        dataset = dataset.reshape(dataset.shape[0], 3, 32, 32)
        dataset = np.moveaxis(dataset, 1, 3)
        return dataset, label

    def Q5_1(self):
        plt.figure("Show image", figsize=(6,6))  
        for i in range(1,10):
            rand = np.random.randint(low=0, high=self.x_test.shape[0])
            plt.subplot(3, 3, i)
            plt.title(self.label_name[self.y_test_show[rand]])
            plt.axis('off')
            plt.imshow(self.x_test[rand], interpolation='nearest') 
        plt.show()
    
    def Q5_2(self):
        print('\nHyperparameters:\nBatch size: %s\nEpochs: %s\nLearning rate: %s\nOptimizer: %s' 
                                            %(self.batch_size, self.epochs, self.lr, 'Adam'))

    def Q5_3(self):
        self.model.summary()
    
    def Q5_4(self):
        history_image = plt.imread("./Dataset_CvDl_Hw1/cifar-10-python/model_history.png")
        plt.figure("Training Process", figsize=(10,5))  
        plt.imshow(history_image)
        plt.show()

    def Q5_5(self, selected_num=100):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.opt,
                           metrics=['accuracy'])
        x = self.x_test[selected_num]
        target = np.expand_dims(x, axis=0) # 必須轉成 batch 的樣子: (1, 32, 32, 3)
        prediction_class = np.argmax(self.model.predict(target), axis=-1) # 找預測中機率的最大值
        plt.figure("Prediction", figsize=(10,5))  
        # 畫圖 + 顯示預測的種類
        plt.subplot(1, 2, 1)
        plt.title("Predicted as: "+str(self.label_name[prediction_class[0]])) # 因為回傳的是 list
        plt.imshow(x)
        # 畫直方圖 + 設定標籤
        plt.subplot(1, 2, 2)
        plt.bar(self.label_name, self.model.predict(target)[0], width=0.8)
        plt.xticks(rotation='vertical') 
        plt.show()
        


if __name__ == "__main__":
    q5 = Q5()
    q5.Q5_1()
    q5.Q5_3()
    q5.Q5_4()
    q5.Q5_5()