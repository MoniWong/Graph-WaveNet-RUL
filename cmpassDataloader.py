from cmapssdata import CMAPSSDataset
import numpy as np


class data():
    def __init__(self, batches, data, label, batchSize ):
        self.batches = batches
        self.trainDataFeature = data
        self.trainDataLabel = label
        self.batchSize = batchSize
        self.batchpoint = 0

    def nextBatch(self):
        if self.batchpoint >= self.batches - 1:
            self.batchpoint = 0
            return self.trainDataFeature[(self.batches - 1) * self.batchSize: self.batches * self.batchSize, :, :], \
                   self.trainDataLabel[(self.batches - 1) * self.batchSize: self.batches * self.batchSize]


        else:
            self.batchpoint = self.batchpoint + 1
            return self.trainDataFeature[(self.batchpoint - 1) * self.batchSize:self.batchpoint * self.batchSize, :, :], \
                   self.trainDataLabel[(self.batchpoint - 1) * self.batchSize:self.batchpoint * self.batchSize]


class dataLoader():
    def __init__(self, bs = 10, sl = 50, fd_number = '1'):
        self.batchSize = bs
        self.sequence_length = sl
        self.cmpass = CMAPSSDataset(fd_number=fd_number, batch_size=bs, sequence_length=sl)
        self.Data = self.cmpass.get_train_data()
        self.adj_mx = self.cmpass.get_adj_mx()
        DataFeature = self.cmpass.get_feature_slice(self.Data)
        DataLabel = self.cmpass.get_label_slice(self.Data)
        # self.trainEngineID = cmpass.get_engine_id(trainData)
        self.Data1 = self.cmpass.get_test_data()
        DataFeature1 = self.cmpass.get_feature_slice(self.Data1)
        DataLabel1 = self.cmpass.get_label_slice(self.Data1)
        # self.trainEngineID1 = cmpass.get_engine_id(testData)
        self.DataFeature = np.concatenate((DataFeature, DataFeature1), axis = 0)
        self.DataLabel =  np.concatenate((DataLabel, DataLabel1), axis = 0)

        np.random.seed(0)
        index = np.random.permutation(self.DataFeature.shape[0])
        self.DataFeature = self.DataFeature[index,:,:]
        self.DataLabel = self.DataLabel[index]

        leng = self.DataFeature.shape[0]
        self.trainDataFeature = self.DataFeature[0:int(leng*0.7),:,:]
        self.trainDataLabel = self.DataLabel[0:int(leng*0.7)]
        self.train_batches = self.trainDataFeature.shape[0]//self.batchSize

        #self.batches = self.trainDataFeature.shape[0]//self.batchSize
        self.testDataFeature = self.DataFeature[int(leng*0.7):,:,:]
        self.testDataLabel = self.DataLabel[int(leng*0.7):]
        self.test_batches = self.testDataFeature.shape[0]//self.batchSize


        self.valiData = self.DataFeature[int(leng*0.9):,:,:]
        self.valiLabel = self.DataLabel[int(leng*0.9):]
        self.vali_batches = self.valiData.shape[0]//self.batchSize
        # self.batchpoint = 0
        self.trainLoader = data(self.train_batches, self.trainDataFeature, self.trainDataLabel, self.batchSize)
        self.testLoader  = data(self.test_batches, self.testDataFeature, self.testDataLabel, self.batchSize)
        self.valiLoader = data(self.vali_batches, self.valiData, self.valiLabel, self.batchSize)

        #将测试集和训练集的数据随机混合，70%作为训练数据，30%作为测试数据
        # self.test_batches = self.testDataFeature.shape[0]//self.bacthSize

        #self.batchpoint = 0
    def get_one_piece(self, num):
        return self.cmpass.get_one_piece(num, self.Data1)
    
    def get_testBatch(self):
        return self.testDataFeature, self.testDataLabel



if __name__ == '__main__':
    d = dataLoader(64,50)
    z , v= d.nextBatch()
    co = 1






