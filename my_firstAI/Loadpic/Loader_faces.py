from PIL import Image
import numpy as np

class DataLoader:
    images = []
    labels = []
    labels_onehot = []

    dataPath = '/home/thu-skyworks/shejp20/python/pytorch_study/Mnist_sample/Dataset/'

    train_images = []
    train_labels = []
    train_labels_onehot = []
    
    validation_images = []
    validation_labels = []
    validation_labels_onehot = []

    test_images = []
    test_labels = []
    test_labels_onehot = []

    dataCount = 200

    #加载Mnist数据
    def loadMnistData(self):

        for j in range(self.dataCount): #每个标签下有dataCount个数据
            for i in range(10): #分别加载0-9标签的数据
                picPath = self.dataPath + str(i) + '/' + str(j) + '.jpg'
                image = self.loadPicArray(picPath)
                label = i
                self.images.append(image)
                self.labels.append(label)
        self.labels_onehot = np.eye(10)[self.labels] #根据所有数据的标签值直接得到所有数据的标签的onehot形式

        #打乱数据，使用相同的次序打乱images、labels和labels_onehot，保证数据仍然对应
        state = np.random.get_state()
        np.random.shuffle(self.images)
        np.random.set_state(state)
        np.random.shuffle(self.labels)
        np.random.set_state(state)
        np.random.shuffle(self.labels_onehot)

        #按比例切割数据，分为训练集、验证集和测试集
        trainIndex = int(self.dataCount * 10 * 0.65)
        validationIndex = int(self.dataCount * 10 * 0.85)
        self.train_images = self.images[0 : trainIndex]
        self.train_labels = self.labels[0 : trainIndex]
        self.train_labels_onehot = self.labels_onehot[0 : trainIndex]
        self.validation_images = self.images[trainIndex : validationIndex]
        self.validation_labels = self.labels[trainIndex : validationIndex]
        self.validation_labels_onehot = self.labels_onehot[trainIndex : validationIndex]
        self.test_images = self.images[validationIndex : ]
        self.test_labels = self.labels[validationIndex : ]
        self.test_labels_onehot = self.labels_onehot[validationIndex : ]


    #读取图片数据，得到图片对应的像素值的数组，均一化到0-1之前
    def loadPicArray(self, picFilePath):
        picData = Image.open(picFilePath)
        picArray = np.array(picData).flatten() / 255.0
        return picArray

    def getTrainData(self):
        return self.train_images, self.train_labels, self.train_labels_onehot

    def getValidationData(self):
        return self.validation_images, self.validation_labels, self.validation_labels_onehot

    def getTestData(self):
        return self.test_images, self.test_labels, self.test_labels_onehot