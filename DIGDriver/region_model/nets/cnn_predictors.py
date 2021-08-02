import torch
from torch import nn, transpose
from torch.autograd import Variable
from torch.nn import functional as F


class FCNet(nn.Module):
    def __init__(self, shape, task_num):
        super(FCNet, self).__init__()
        print('Intializing FCNet...')
        self.inp_len = shape[1]
        self.inp_size = shape[2]
        self.task_num = task_num

        self.hidden_dim_1 = 128
        self.hidden_dim_2 = 16

        self.fc1_lst = nn.ModuleList()
        self.fc2_lst = nn.ModuleList()
        self.fc3_lst = nn.ModuleList()
        for _ in range(self.task_num):
            self.fc1_lst.append(nn.Linear(in_features=self.inp_size, out_features=self.hidden_dim_1))
            self.fc2_lst.append(nn.Linear(in_features=self.hidden_dim_1, out_features=self.hidden_dim_2))
            self.fc3_lst.append(nn.Linear(in_features=self.hidden_dim_2, out_features=1))

    def forward(self, x: Variable) -> (Variable):
        if self.inp_len > 1:
            x = x.mean(dim=1)

        outputs = []
        feature_vecs = []
        for i in range(self.task_num):
            x = F.relu(self.fc1_lst[i](x))
            x = F.relu(self.fc2_lst[i](x))
            feature_vecs.append(x)
            outputs.append(self.fc3_lst[i](x).reshape(-1))

        return outputs, feature_vecs, None


class AutoregressiveFCNet(nn.Module):
    def __init__(self, shape, task_num):
        super(AutoregressiveFCNet, self).__init__()
        print('Intializing AutoregressiveFCNet...')
        self.inp_len = shape[1]
        self.inp_size = shape[2] + 2
        self.task_num = task_num

        self.hidden_dim_1 = 128
        self.hidden_dim_2 = 16

        self.fc1_lst = nn.ModuleList()
        self.fc2_lst = nn.ModuleList()
        self.fc3_lst = nn.ModuleList()
        for _ in range(self.task_num):
            self.fc1_lst.append(nn.Linear(in_features=self.inp_size, out_features=self.hidden_dim_1))
            self.fc2_lst.append(nn.Linear(in_features=self.hidden_dim_1, out_features=self.hidden_dim_2))
            self.fc3_lst.append(nn.Linear(in_features=self.hidden_dim_2, out_features=1))

    def forward(self, x, auto_x):
        if self.inp_len > 1:
            x = x.mean(dim=1)

        outputs = []
        feature_vecs = []
        for i in range(self.task_num):
            #task_x = torch.cat([torch.zeros(x.size()).cuda(), auto_x], 1)
            task_x = torch.cat([x, auto_x], 1)  # adding autoregressive features
            task_x = F.relu(self.fc1_lst[i](task_x))
            task_x = F.relu(self.fc2_lst[i](task_x))
            feature_vecs.append(task_x)
            outputs.append(self.fc3_lst[i](task_x).reshape(-1))

        return outputs, feature_vecs, None


class SimpleMultiTaskResNet(nn.Module):
    def __init__(self, shape, task_num, get_attention_maps=False):
        super(SimpleMultiTaskResNet, self).__init__()
        print('Intializing SimpleMultiTaskResNet...')
        self.get_attention_maps = get_attention_maps
        self.inp_len = shape[1]
        self.inp_size = shape[2]
        self.task_num = task_num

        self.hidden_dim = 128
        self.fc2_dim = 128
        self.fc3_dim = 16

        if self.get_attention_maps:
            self.att_conv1 = nn.Conv1d(in_channels=self.inp_size, out_channels=self.inp_size, kernel_size=5, padding=2, stride=1)
            #self.att_bn1 = nn.BatchNorm1d(self.inp_size)
            self.att_conv2 = nn.Conv1d(in_channels=self.inp_size, out_channels=self.inp_size, kernel_size=3, padding=1, stride=1)
            #self.att_bn2 = nn.BatchNorm1d(self.inp_size)

        self.conv11 = nn.Conv1d(in_channels=self.inp_size, out_channels=128, kernel_size=5, padding=1, stride=1)
        self.bn11 = nn.BatchNorm1d(128)
        self.conv12 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.bn12 = nn.BatchNorm1d(256)

        self.conv21 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.bn21 = nn.BatchNorm1d(256)
        self.conv22 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.bn22 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm1d(512)

        self.conv41 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.bn41 = nn.BatchNorm1d(512)
        self.conv42 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.bn42 = nn.BatchNorm1d(512)

        self.conv5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv61 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1)
        self.bn61 = nn.BatchNorm1d(1024)
        self.conv62 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1)
        self.bn62 = nn.BatchNorm1d(1024)

        self.fc1_lst = nn.ModuleList()
        self.fc2_lst = nn.ModuleList()
        self.fc3_lst = nn.ModuleList()
        for _ in range(self.task_num):
            self.fc1_lst.append(nn.Linear(in_features=int(1024 * 13), out_features=self.fc2_dim))
            self.fc2_lst.append(nn.Linear(in_features=self.fc2_dim, out_features=self.fc3_dim))
            self.fc3_lst.append(nn.Linear(in_features=self.fc3_dim, out_features=1))

    def forward(self, x: Variable) -> (Variable):
        x = transpose(x, 1, 2)

        if self.get_attention_maps:
            #att_x = F.relu(self.att_bn1(self.att_conv1(x)))
            #att_x = F.relu(self.att_bn2(self.att_conv2(att_x)))
            att_x = F.relu(self.att_conv1(x))
            att_x = F.relu(self.att_conv2(att_x))
            att_x = F.softmax(att_x, dim=2)
            #att_x = torch.sigmoid(att_x) #* F.softmax(att_x.mean(dim=1).unsqueeze(1), dim=2).expand_as(x)
            #x = x * att_x.expand_as(x)
            x = x * att_x

        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        res = x
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x += res
        x = F.relu(self.bn3(self.conv3(x)))
        res = x
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x += res
        x = F.relu(self.bn5(self.conv5(x)))
        res = x
        x = F.relu(self.bn61(self.conv61(x)))
        x = F.relu(self.bn62(self.conv62(x)))
        x += res

        x = x.view(-1, int(1024 * 13))

        outputs = []
        feature_vecs = []
        for i in range(self.task_num):
            task_x = F.relu(self.fc1_lst[i](x))
            task_x = F.relu(self.fc2_lst[i](task_x))
            feature_vecs.append(task_x)
            outputs.append(self.fc3_lst[i](task_x).reshape(-1))

        if self.get_attention_maps: return outputs, feature_vecs, att_x
        return outputs, feature_vecs, None


class MultiTaskCNN(nn.Module):
    def __init__(self, shape, task_num):
        super(SimpleMultiTaskResNet, self).__init__()
        print('Intializing MultiTaskCNN...')
        self.inp_len = shape[1]
        self.inp_size = shape[2]
        self.task_num = task_num

        self.hidden_dim = 128
        self.fc2_dim = 128

        self.conv_base = nn.Conv1d(in_channels=self.inp_size, out_channels=128, kernel_size=5, padding=3, stride=2)

        self.bn11 = nn.BatchNorm1d(128)
        self.conv11 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=2, stride=2)
        self.bn12 = nn.BatchNorm1d(256)
        self.conv12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.w1 = nn.Linear(in_features=128 * 51, out_features=256 * 27)

        self.bn21 = nn.BatchNorm1d(256)
        self.conv21 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=2, stride=2)
        self.bn22 = nn.BatchNorm1d(256)
        self.conv22 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.w2 = nn.Linear(in_features=256 * 27, out_features=256 * 15)

        self.bn31 = nn.BatchNorm1d(256)
        self.conv31 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=2, stride=2)
        self.bn32 = nn.BatchNorm1d(512)
        self.conv32 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.w3 = nn.Linear(in_features=256 * 15, out_features=512 * 9)

        self.bn41 = nn.BatchNorm1d(512)
        self.conv41 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=2, stride=2)
        self.bn42 = nn.BatchNorm1d(1024)
        self.conv42 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1)
        self.w4 = nn.Linear(in_features=512 * 9, out_features=1024 * 6)

        self.bn51 = nn.BatchNorm1d(1024)
        self.conv51 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=2)
        self.bn52 = nn.BatchNorm1d(1024)
        self.conv52 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1)
        self.w5 = nn.Linear(in_features=1024 * 6, out_features=1024 * 3)

        self.fc1_lst = nn.ModuleList()
        self.fc2_lst = nn.ModuleList()
        for _ in range(self.task_num):
            self.fc1_lst.append(nn.Linear(in_features=int(1024 * 3), out_features=self.fc2_dim))
            self.fc2_lst.append(nn.Linear(in_features=self.fc2_dim, out_features=1))

    def forward(self, x: Variable) -> (Variable):
        x = F.relu(self.conv_base(transpose(x, 1, 2)))

        res = x.view(-1, 128 * 51)
        x = self.conv11(F.relu(self.bn11(x)))
        x = self.conv12(F.relu(self.bn12(x)))
        x += self.w1(res).view(-1, 256, 27)

        res = x.view(-1, 256 * 27)
        x = self.conv21(F.relu(self.bn21(x)))
        x = self.conv22(F.relu(self.bn22(x)))
        x += self.w2(res).view(-1, 256, 15)

        res = x.view(-1, 256 * 15)
        x = self.conv31(F.relu(self.bn31(x)))
        x = self.conv32(F.relu(self.bn32(x)))
        x += self.w3(res).view(-1, 512, 9)

        res = x.view(-1, 512 * 9)
        x = self.conv41(F.relu(self.bn41(x)))
        x = self.conv42(F.relu(self.bn42(x)))
        x += self.w4(res).view(-1, 1024, 6)

        res = x.view(-1, 1024 * 6)
        x = self.conv51(F.relu(self.bn51(x)))
        x = self.conv52(F.relu(self.bn52(x)))
        x += self.w5(res).view(-1, 1024, 3)

        x = x.view(-1, int(1024 * 3))

        outputs = []
        for i in range(self.task_num):
            task_x = F.relu(self.fc1_lst[i](x))
            outputs.append(self.fc2_lst[i](task_x).reshape(-1))
        return outputs


class AutoregressiveMultiTaskResNet(nn.Module):
    def __init__(self, shape, task_num, get_attention_maps=False):
        super(AutoregressiveMultiTaskResNet, self).__init__()
        print('Intializing AutoregressiveMultiTaskResNet...')
        self.get_attention_maps = get_attention_maps
        self.inp_len = shape[1]
        self.inp_size = shape[2]
        self.task_num = task_num

        self.hidden_dim = 128
        self.fc2_dim = 128
        self.fc3_dim = 16

        if self.get_attention_maps:
            self.att_conv1 = nn.Conv1d(in_channels=self.inp_size, out_channels=self.inp_size, kernel_size=5, padding=2, stride=1)
            self.att_conv2 = nn.Conv1d(in_channels=self.inp_size, out_channels=self.inp_size, kernel_size=3, padding=1, stride=1)

        self.conv11 = nn.Conv1d(in_channels=self.inp_size, out_channels=128, kernel_size=5, padding=1, stride=1)
        self.bn11 = nn.BatchNorm1d(128)
        self.conv12 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.bn12 = nn.BatchNorm1d(256)

        self.conv21 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.bn21 = nn.BatchNorm1d(256)
        self.conv22 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.bn22 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm1d(512)

        self.conv41 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.bn41 = nn.BatchNorm1d(512)
        self.conv42 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.bn42 = nn.BatchNorm1d(512)

        self.conv5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv61 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1)
        self.bn61 = nn.BatchNorm1d(1024)
        self.conv62 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1)
        self.bn62 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(in_features=int(1024 * 13) + 2 * self.task_num, out_features=self.fc2_dim)
        
        #self.fc1_lst = nn.ModuleList()
        self.fc2_lst = nn.ModuleList()
        self.fc3_lst = nn.ModuleList()
        for _ in range(self.task_num):
            #self.fc1_lst.append(nn.Linear(in_features=int(1024 * 13) + 2, out_features=self.fc2_dim))
            self.fc2_lst.append(nn.Linear(in_features=self.fc2_dim, out_features=self.fc3_dim))
            self.fc3_lst.append(nn.Linear(in_features=self.fc3_dim, out_features=1))

    def forward(self, x, auto_x):
        x = transpose(x, 1, 2)

        if self.get_attention_maps:
            #att_x = F.relu(self.att_bn1(self.att_conv1(x)))
            #att_x = F.relu(self.att_bn2(self.att_conv2(att_x)))
            att_x = F.relu(self.att_conv1(x))
            att_x = F.relu(self.att_conv2(att_x))
            att_x = F.softmax(att_x, dim=2)
            #att_x = torch.sigmoid(att_x) #* F.softmax(att_x.mean(dim=1).unsqueeze(1), dim=2).expand_as(x)
            #x = x * att_x.expand_as(x)
            x = x * att_x

        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        res = x
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x += res
        x = F.relu(self.bn3(self.conv3(x)))
        res = x
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x += res
        x = F.relu(self.bn5(self.conv5(x)))
        res = x
        x = F.relu(self.bn61(self.conv61(x)))
        x = F.relu(self.bn62(self.conv62(x)))
        x += res

        x = x.view(-1, int(1024 * 13))
        ar_x = torch.cat([x, auto_x], dim=1)  # adding autoregressive features
        ar_x = F.relu(self.fc1(ar_x))

        outputs = []
        feature_vecs = []
        for i in range(self.task_num):
            task_x = F.relu(self.fc2_lst[i](ar_x))

            #task_x = torch.cat([torch.zeros(x.size()).cuda(), auto_x], 1)

            #task_x = torch.cat([x, auto_x], dim=1)  # adding autoregressive features
            #task_x = F.relu(self.fc1_lst[i](task_x))
            #task_x = F.relu(self.fc2_lst[i](task_x))
            feature_vecs.append(task_x)
            outputs.append(self.fc3_lst[i](task_x).reshape(-1))

        if self.get_attention_maps: return outputs, feature_vecs, att_x
        return outputs, feature_vecs, None
