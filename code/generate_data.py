import os, sys
sys.path.append(os.getcwd())
import random
random.seed(1)
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(1)
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def obtainfeatures(data,file_path1,strs):
    host_features=[]
    for i in data:
        host_features.append(np.loadtxt(file_path1+i+strs).tolist())
    return np.array(host_features)

class Generator(nn.Module):
    
    def __init__(self,shape1):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(shape1, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, shape1),
        )
        self.main = main

    def forward(self, noise, real_data):
        if FIXED_GENERATOR:
            return noise + real_data
        else:
            output = self.main(noise)
            return output

class Discriminator(nn.Module):

    def __init__(self,shape1):
        super(Discriminator, self).__init__()

        self.fc1=nn.Linear(shape1, 512)
        self.relu=nn.LeakyReLU(0.2)
        self.fc2=nn.Linear(512, 256)
        self.relu=nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, inputs):
        out=self.fc1(inputs)
        out=self.relu(out)
        out=self.fc2(out)
        out=self.relu(out)
        out=self.fc3(out)
        out=self.relu(out)
        out=self.fc4(out)
        return out.view(-1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def inf_train_gen(datas):
    with open("../result/result_GAN/"+datas+".txt") as f:
        MatrixFeaturesPositive = [list(x.split(" ")) for x in f]
    FeaturesPositive = [line[:] for line in MatrixFeaturesPositive[:]]
    dataset2 = np.array(FeaturesPositive, dtype='float32')
    return dataset2
  
def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates) 
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

data=pd.read_csv('../data/data_pos_neg.txt',header=None,sep=',')
data=data[data[2]==1].values.tolist()
phage=[i[0] for i in data]
host=[i[1] for i in data]
phage_feature_pro=obtainfeatures(phage,'../data/phage_protein_normfeatures/','.txt')
phage_feature_dna=obtainfeatures(phage,'../data/phage_dna_norm_features/','.txt')
host_feature_pro=obtainfeatures(host,'../data/host_protein_normfeatures/','.txt')
host_feature_dna=obtainfeatures(host,'../data/host_dna_norm_features/','.txt')
phage_all=np.concatenate((phage_feature_dna, phage_feature_pro),axis=1)
host_all=np.concatenate((host_feature_dna, host_feature_pro),axis=1)
###save features of real positive samples
if not os.path.exists('../result/'):
    os.mkdir('../result')
    os.mkdir('../result/result_GAN')
np.savetxt('../result/result_GAN/data_GAN.txt',np.concatenate((phage_all, host_all),axis=1))
data = inf_train_gen('data_GAN')
FIXED_GENERATOR = False  
LAMBDA = .1  
CRITIC_ITERS = 5  
BATCH_SIZE = len(data)
ITERS = 100000
use_cuda = False
netG = Generator(data.shape[1])
netD = Discriminator(data.shape[1])
netD.apply(weights_init)
netG.apply(weights_init)
if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()
optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
one = torch.tensor(1, dtype=torch.float)  ###torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()
###iteration process
for iteration in range(ITERS):
    for p in netD.parameters():  
        p.requires_grad = True  
    data = inf_train_gen('data_GAN')
    real_data = torch.FloatTensor(data)
    if use_cuda:
        real_data = real_data.cuda()
    real_data_v = autograd.Variable(real_data)
    noise = torch.randn(BATCH_SIZE, data.shape[1])
    if use_cuda:
        noise = noise.cuda()
    noisev = autograd.Variable(noise, volatile=True)  
    fake = autograd.Variable(netG(noisev, real_data_v).data)
    fake_output=fake.data.cpu().numpy()
    for iter_d in range(CRITIC_ITERS):
        netD.zero_grad()
        D_real = netD(real_data_v)
        D_real = D_real.mean()
        D_real.backward(mone)
        noise = torch.randn(BATCH_SIZE, data.shape[1])
        if use_cuda:
            noise = noise.cuda()
        noisev = autograd.Variable(noise, volatile=True)  
        fake = autograd.Variable(netG(noisev, real_data_v).data)        
        inputv = fake
        D_fake = netD(inputv)
        D_fake = D_fake.mean()
        D_fake.backward(one)
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        gradient_penalty.backward()
        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()
    ###save generated sample features every 200 iteration
    if iteration%200 == 0:
        fake_writer = open("../result/result_GAN/Iteration_"+str(iteration)+".txt","w")
        for rowIndex in range(len(fake_output)):
            for columnIndex in range(len(fake_output[0])):
                fake_writer.write(str(fake_output[rowIndex][columnIndex]) + ",")
            fake_writer.write("\n")
        fake_writer.flush()
        fake_writer.close()
    if not FIXED_GENERATOR:
        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()
        real_data = torch.Tensor(data)
        if use_cuda:
            real_data = real_data.cuda()
        real_data_v = autograd.Variable(real_data)
        noise = torch.randn(BATCH_SIZE, data.shape[1])
        if use_cuda:
            noise = noise.cuda()
        noisev = autograd.Variable(noise)
        fake = netG(noisev, real_data_v)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()

####test model result, LOOCV to select optimal pseudo samples
with open("../result/result_GAN/data_GAN.txt") as f:
    MatrixFeatures = [list(x.split(" ")) for x in f]
realFeatures = [line[:] for line in MatrixFeatures[:]]
realDataset = np.array(realFeatures, dtype='float32')
# Adding equal numbers of binary labels
label=[]
for rowIndex in range(len(realDataset)):
    label.append(1)
for rowIndex in range(len(realDataset)):
    label.append(0)
labelArray=np.asarray(label)
opt_diff_accuracy_05=0.5
opt_Epoch=0
opt_accuracy=0
allresult=[]
for indexEpoch in range(500):
    epoch = indexEpoch * 200
    with open("../result/result_GAN/Iteration_"+str(epoch)+".txt") as f:
          MatrixFeatures = [list(x.split(",")) for x in f]
    fakeFeatures = [line[:-1] for line in MatrixFeatures[:]]
    fakedataset = np.array(fakeFeatures, dtype='float32')
    realFakeFeatures=np.vstack((realDataset, fakedataset))

    prediction_list=[]
    real_list=[]
    ####LOOCV
    loo = LeaveOneOut()
    loo.get_n_splits(realFakeFeatures)
    for train_index, test_index in loo.split(realFakeFeatures):
        X_train, X_test = realFakeFeatures[train_index], realFakeFeatures[test_index]
        y_train, y_test = labelArray[train_index], labelArray[test_index]
        knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
        predicted_y = knn.predict(X_test)
        prediction_list.append(predicted_y)
        real_list.append(y_test)
    accuracy=accuracy_score(real_list, prediction_list)
    allresult.append(str(indexEpoch)+"%"+str(accuracy))
    diff_accuracy_05=abs(accuracy-0.5)
    if diff_accuracy_05 < opt_diff_accuracy_05:
        opt_diff_accuracy_05=diff_accuracy_05
        opt_Epoch=epoch
        opt_accuracy=accuracy
print(str(opt_Epoch)+"%"+str(opt_accuracy))

