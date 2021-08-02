import torch
from torch import nn, transpose
from torch.autograd import Variable
from torch.nn import functional as F

class ResNetEncoder(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.inp_len = shape[1]
        self.inp_size = shape[2]

        self.hidden_dim = 128
        self.fc2_dim = 128
        self.fc3_dim = 16


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

        self.fc1 = nn.Linear(in_features=int(1024 * 13), out_features=self.fc2_dim)
        self.fc2 = nn.Linear(in_features=self.fc2_dim, out_features=self.fc3_dim)
        self.fc3 = nn.Linear(in_features=self.fc3_dim, out_features=16)

        #decoding network
        self.dfc3 = nn.Linear(in_features=16, out_features=self.fc3_dim)
        self.dfc2 = nn.Linear(in_features=self.fc3_dim, out_features=self.fc2_dim)
        self.dfc1 = nn.Linear(in_features=self.fc2_dim, out_features=int(1024 * 13))

    def forward(self, x):
        x = transpose(x, 1, 2)

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

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ResNet_NoBN_Encoder(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.inp_len = shape[1]
        self.inp_size = shape[2]

        self.hidden_dim = 128
        self.fc2_dim = 128
        self.fc3_dim = 16


        self.conv11 = nn.Conv1d(in_channels=self.inp_size, out_channels=128, kernel_size=5, padding=1, stride=1)
        self.conv12 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)

        self.conv21 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv22 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2)

        self.conv41 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv42 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)

        self.conv5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2)

        self.conv61 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1)
        self.conv62 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1)

        self.fc1 = nn.Linear(in_features=int(1024 * 13), out_features=self.fc2_dim)
        self.fc2 = nn.Linear(in_features=self.fc2_dim, out_features=self.fc3_dim)
        self.fc3 = nn.Linear(in_features=self.fc3_dim, out_features=16)

    def forward(self, x):
        x = transpose(x, 1, 2)

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        res = x
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x += res
        x = F.relu(self.conv3(x))
        res = x
        x = F.relu(self.conv41(x))
        x = F.relu(self.conv42(x))
        x += res
        x = F.relu(self.conv5(x))
        res = x
        x = F.relu(self.conv61(x))
        x = F.relu(self.conv62(x))
        x += res

        x = x.view(-1, int(1024 * 13))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ResNetDecoder(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.inp_len = shape[1]
        self.inp_size = shape[2]

        self.hidden_dim = 128
        self.fc2_dim = 128
        self.fc3_dim = 16

        #decoding network
        self.dfc3 = nn.Linear(in_features=16, out_features=self.fc3_dim)
        self.dfc2 = nn.Linear(in_features=self.fc3_dim, out_features=self.fc2_dim)
        self.dfc1 = nn.Linear(in_features=self.fc2_dim, out_features=int(1024 * 13))

        #self.dbn62 = nn.BatchNorm1d(1024)
        self.dconv62 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1)
        #self.dbn61 = nn.BatchNorm1d(1024)
        self.dconv61 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1)

        #self.dbn5 = nn.BatchNorm1d(1024)
        self.dconv5 = nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, stride=2)

        #self.dbn42 = nn.BatchNorm1d(512)
        self.dconv42 = nn.ConvTranspose1d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        #self.dbn41 = nn.BatchNorm1d(512)
        self.dconv41 = nn.ConvTranspose1d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)

        #self.dbn3 = nn.BatchNorm1d(512)
        self.dconv3 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=2)

        #self.dbn22 = nn.BatchNorm1d(256)
        self.dconv22 = nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        #self.dbn21 = nn.BatchNorm1d(256)
        self.dconv21 = nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)

        #self.dbn12 = nn.BatchNorm1d(256)
        self.dconv12 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=2)
        #self.dbn11 = nn.BatchNorm1d(128)
        self.dconv11 = nn.ConvTranspose1d(in_channels=128, out_channels=self.inp_size, kernel_size=5, padding=1, stride=1)

        self.last = torch.nn.ConvTranspose1d(in_channels = self.inp_size, out_channels = self.inp_size, kernel_size = 4, padding = 1, stride =1)

    def forward(self, x):
        x = F.relu(self.dfc3(x))
        x = F.relu(self.dfc2(x))
        x = F.relu(self.dfc1(x))

        x = x.view(-1, 1024, 13)

        x = F.relu(self.dconv62(x))
        x = F.relu(self.dconv61(x))

        x = F.relu(self.dconv5(x))

        x = F.relu(self.dconv42(x))
        x = F.relu(self.dconv41(x))

        x = F.relu(self.dconv3(x))

        x = F.relu(self.dconv22(x))
        x = F.relu(self.dconv21(x))

        x = F.relu(self.dconv12(x))
        x = F.relu(self.dconv11(x))
        x = F.relu(self.last(x))
        x = transpose(x, 1, 2)
        return x

class ResNetLinearDecoder(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.inp_len = shape[1]
        self.inp_size = shape[2]

        self.hidden_dim = 128
        self.fc2_dim = 128
        self.fc3_dim = 16

        #decoding network
        self.decoder = nn.Sequential(
        nn.Linear(in_features=16, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2048),
        nn.ReLU(),
        nn.Linear(in_features=2048, out_features=self.inp_len * self.inp_size),
        )

    def forward(self, x):
        x = self.decoder(x)
        x = x.view(-1, 100, 734)
        return x

class ResNetShallowLinearDecoder(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.inp_len = shape[1]
        self.inp_size = shape[2]

        self.hidden_dim = 128
        self.fc2_dim = 128
        self.fc3_dim = 16

        #decoding network
        self.decoder = nn.Sequential(
        nn.Linear(in_features=16, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=self.inp_len * self.inp_size),
        )

    def forward(self, x):
        x = self.decoder(x)
        x = x.view(-1, 100, 734)
        return x

class ResNetAE(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.inp_len = shape[1]
        self.inp_size = shape[2]

        self.hidden_dim = 128
        self.fc2_dim = 128
        self.fc3_dim = 16

        self.encoder = ResNetEncoder(shape)
        self.decoder = ResNetDecoder(shape)

    def forward(self, x):
        encoded = self.encoder(x)
        x = self.decoder(encoded)
        return encoded, x

    def embeding(self, x):
        x = self.encoder(x)
        return x
class ResNetAE_LD(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.inp_len = shape[1]
        self.inp_size = shape[2]

        self.hidden_dim = 128
        self.fc2_dim = 128
        self.fc3_dim = 16

        self.encoder = ResNetEncoder(shape)
        self.decoder = ResNetLinearDecoder(shape)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def embeding(self, x):
        x = self.encoder(x)
        return x

class ResNetAE_SLD(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.inp_len = shape[1]
        self.inp_size = shape[2]

        self.hidden_dim = 128
        self.fc2_dim = 128
        self.fc3_dim = 16

        self.encoder = ResNetEncoder(shape)
        self.decoder = ResNetShallowLinearDecoder(shape)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def embeding(self, x):
        x = self.encoder(x)
        return x
