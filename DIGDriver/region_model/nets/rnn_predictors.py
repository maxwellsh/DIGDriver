from torch import nn, transpose
from torch.autograd import Variable
from torch.nn import functional as F


class MultiTaskLinear(nn.Module):
    def __init__(self, shape, task_num):
        super(MultiTaskLinear, self).__init__()
        self.inp_len = shape[1]
        self.inp_size = shape[2]
        self.task_num = task_num

        self.hidden_dim = 128
        self.fc2_dim = 128

        self.conv1 = nn.Conv1d(in_channels=self.inp_size, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm1d(256)

        self.birnn = nn.LSTM(input_size=256, hidden_size=self.hidden_dim, num_layers=3, batch_first=True, bidirectional=True)

        self.fc1_lst = nn.ModuleList()
        self.fc2_lst = nn.ModuleList()
        for _ in range(self.task_num):
            self.fc1_lst.append(nn.Linear(in_features=int(self.hidden_dim * 2), out_features=self.fc2_dim))
            self.fc2_lst.append(nn.Linear(in_features=self.fc2_dim, out_features=1))

    def forward(self, x: Variable) -> (Variable):
        self.birnn.flatten_parameters()
        x = self.bn1(F.relu(self.conv1(transpose(x, 1, 2))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.birnn(transpose(x, 1, 2))

        outputs = []
        for i in range(self.task_num):
            task_x = F.relu(self.fc1_lst[i](x[0][:, -1, :]))
            outputs.append(self.fc2_lst[i](task_x).reshape(-1))

        return outputs
    
class MultiTaskRNN(nn.Module):
    def __init__(self, shape, task_num):
        super(MultiTaskRNN, self).__init__()
        self.inp_len = shape[1]
        self.inp_size = shape[2]
        self.task_num = task_num

        self.hidden_dim = 128
        self.fc2_dim = 128

        self.conv1 = nn.Conv1d(in_channels=self.inp_size, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm1d(256)

        self.rnn_lst = nn.ModuleList()
        self.fc1_lst = nn.ModuleList()
        self.fc2_lst = nn.ModuleList()
        for _ in range(self.task_num):
            self.rnn_lst.append(nn.LSTM(input_size=256, hidden_size=self.hidden_dim, num_layers=3, batch_first=True, bidirectional=True))
            self.fc1_lst.append(nn.Linear(in_features=int(self.hidden_dim * 2), out_features=self.fc2_dim))
            self.fc2_lst.append(nn.Linear(in_features=self.fc2_dim, out_features=1))

    def forward(self, x: Variable) -> (Variable):
        x = self.bn1(F.relu(self.conv1(transpose(x, 1, 2))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        outputs = []
        for i in range(self.task_num):
            self.rnn_lst[i].flatten_parameters()
            task_x = self.rnn_lst[i](transpose(x, 1, 2))
            task_x = F.relu(self.fc1_lst[i](task_x[0][:, -1, :]))
            outputs.append(self.fc2_lst[i](task_x).reshape(-1))

        return outputs

class MultiTaskHierarchicalLinear(nn.Module):
    def __init__(self, shape, task_num):
        super(MultiTaskHierarchicalLinear, self).__init__()
        self.inp_len = shape[1]
        self.inp_size = shape[2]
        self.task_num = task_num

        self.hidden_dim = 128
        self.fc2_dim = 128

        self.conv1 = nn.Conv1d(in_channels=self.inp_size, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm1d(256)

        self.birnn = nn.LSTM(input_size=256, hidden_size=self.hidden_dim, num_layers=3, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(in_features=int(self.hidden_dim * 2), out_features=self.fc2_dim)
        self.t1_out = nn.Linear(in_features=self.fc2_dim, out_features=1)
        
        self.fc2 = nn.Linear(in_features=self.fc2_dim, out_features=self.fc2_dim)
        self.t2_out = nn.Linear(in_features=self.fc2_dim, out_features=1)
                
    def forward(self, x: Variable) -> (Variable):
        self.birnn.flatten_parameters()
        x = self.bn1(F.relu(self.conv1(transpose(x, 1, 2))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.birnn(transpose(x, 1, 2))
        x = F.relu(self.fc1(x[0][:, -1, :]))
        out1 = self.t1_out(x).reshape(-1)

        x = F.relu(self.fc2(x))
        out2 = self.t2_out(x).reshape(-1)

        return [out1, out2]
