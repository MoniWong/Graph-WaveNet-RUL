import torch
import numpy as np
import argparse
import time
import os
import util
# import matplotlib.pyplot as plt
from engine import trainer
from cmpassDataloader import dataLoader
from cmapssdata import CMAPSSDataset
# from cmpassDataloader2 import dataLoader
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cpu',help='')
parser.add_argument('--fd_number',type=str,default='1',help='subset id')
parser.add_argument('--adjtype',type=str,default='transition',help='adj type')
parser.add_argument('--load',type=str,default=r'E:\essay\code\Graph-WaveNet-master\fd1_wavenet\fd1_9.19.pth',help='load path')
parser.add_argument('--gcn_bool',action='store_true',default = True, help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',default = True, help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true', default = False , help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true', default = False ,help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=18,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
args = parser.parse_args()

def main():

    #load data
    device = torch.device(args.device)
    # sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    # supports = [torch.tensor(i).to(device) for i in adj_mx]
    data = dataLoader( bs = args.batch_size, sl = args.seq_length, fd_number =args.fd_number)
    adj_mx = util.load_adj(data.adj_mx.values,args.adjtype)
    #adj_mx = data.adj_mx.values
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None


    engine = trainer(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)
    engine.model.load_state_dict(torch.load(args.load))
    # engine.model.load_state_dict(torch.load(args.load))
    print("start testing...", flush=True)

    val_time = []
    #test

    test_loss = []
    test_asf = []
    test_rmse = []

    s1 = time.time()
    for iter in range(data.test_batches):
        testx, testy  = data.testLoader.nextBatch()
        testx = testx[:, :, :, np.newaxis]
        testx = torch.Tensor(testx).to(device)
        testx = testx.permute((0, 3, 2, 1))
        testy = torch.Tensor(testy).to(device)
        metrics = engine.eval(testx, testy)
        test_loss.append(metrics[0])
        test_rmse.append(metrics[1])
        test_asf.append(metrics[2])
    s2 = time.time()
    val_time.append(s2-s1)


    test_loss = np.mean(test_loss)
    test_rmse = np.mean(test_rmse)
    test_asf = np.mean(test_asf)


    log = 'Test Loss: {:.4f},Test RMSE: {:.4f}, Test_asf:{:.4f}, inference Time: {:.4f}/epoch'
    print(log.format(test_loss, test_rmse, test_asf, (s2 - s1)),flush=True)

def test_sample(sample_num):
    device = torch.device(args.device)
    # sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    # supports = [torch.tensor(i).to(device) for i in adj_mx]
    print(args)
    adjinit = None
    supports = None
    engine = trainer(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)
    engine.model.load_state_dict(torch.load(args.load))
    data = dataLoader( bs = args.batch_size, sl = args.seq_length, fd_number =args.fd_number)
    test_feature, test_label = data.get_one_piece(sample_num)


    test_feature_b = np.reshape(test_feature, [len(test_feature), args.seq_length, 18])
    test_feature_b = test_feature_b[:, :, :, np.newaxis]
    test_feature_b = torch.Tensor(test_feature_b).to(device)
    test_feature_b = test_feature_b.permute((0, 3, 2, 1))
    result = engine.eval2(test_feature_b)
    # bc = len(test_feature)//args.batch_size
    # result = []
    # for i in range(bc):
    #     test_feature_b =  np.reshape(test_feature[bc*args.batch_size, (bc+1)*args.batch_size], [args.batch_size, args.seq_length, -1])
    #     result = result +  engine.model.eval2(test_feature_b)
    p = 1
    leng = len(result)
    plt.plot(range(leng), result, marker='|', color='coral', linewidth=1.0, linestyle='--', label='Prediction')
    plt.plot(range(leng), test_label, linestyle='-', label='Label ')

    plt.xlabel("Time(Cycle)")
    plt.ylabel("RUL(Cycle)")
    plt.title(r"RUL prediction sample")
    plt.legend()
    # plt.savefig(path)
    plt.show()

def polt_feature():
    data = CMAPSSDataset(fd_number='1', batch_size=10, sequence_length=13)
    train = data.train_data
    train = train[train['id'] == 1]
    train = np.array(train)
    for i in range(train.shape[1]):
        p = np.array(train).shape[0]
        plt.plot(range(p), np.array(train[:,i]))
    plt.ylim([-4,4])
    plt.show()


if __name__ == "__main__":
    main()
    #test_sample(2)
    #polt_feature()
    # data = CMAPSSDataset(fd_number='1', batch_size=10, sequence_length=13)
    # train = data.train_data
    # train = train[train['id'] == 1]
    # train = np.array(train)
    # list = [15, 16]
    # for i in range(len(list)):
    #     p = np.array(train).shape[0]
    #     plt.plot(range(p), np.array(train[:,list[i]]), marker='|', linewidth=1.0, linestyle='--', label='sensor' + str(list[i]))
    # plt.ylabel("Sensory Feature")
    # plt.xlabel("Time(Cycle)")
    # plt.title(r"Raw Sensory Data")
    # plt.legend()
    #
    # plt.ylim([-4,4])
    # plt.show()



    w  = 1


    # t1 = time.time()
    # main()
    # t2 = time.time()
    # print("Total time spent: {:.4f}".format(t2-t1))
