import torch.optim as optim
from model import *
import util
class trainer():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=1, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        predict = self.model(input)#output = [batch_size,12,num_nodes,1]
        #predict = output.reshape(output.shape[0],output.shape[2],1)
        #output = output.transpose(1,3) #[batch_size,1,num_nodes,12]
        real = real_val #[batch_size,num_nodes,12]
        #predict = output #逆归一化后的值

        loss = self.loss(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        #mape = util.masked_mape(predict,real).item()
        rmse = util.masked_rmse(predict,real).item()
        score = util.score(predict,real).item()
        return loss.item(),rmse,score

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        predict = self.model(input)
        #predict = output.reshape(output.shape[0],output.shape[2],1)
        #output = [batch_size,12,num_nodes,1]
        #real = torch.unsqueeze(real_val,dim=1)
        real = real_val
        #predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real)
        #mape = util.masked_mape(predict,real).item()
        rmse = util.masked_rmse(predict,real).item()
        score = util.score(predict,real).item()
        return loss.item(),rmse, score

    def eval2(self, input):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        # predict = self.scaler.inverse_transform(output)
        predict = output

        return predict.cpu().detach().numpy()