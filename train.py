import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from utils.dataset import PatternDataset
import numpy as np

W = 100
H = 50

# conditionAL VAE
class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
        # code encoders for conditioning score
        self.fc_code1 = nn.Linear(1,h_dim1)
        self.fc_code2 = nn.Linear(1,h_dim2)
    def encoder(self, x,code):
        h = F.relu(self.fc1(x) + self.fc_code1(code)) 
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z,code):
        h = F.relu(self.fc4(z)+ self.fc_code2(code))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x, score):
        code = score.view(-1,1)
        mu, log_var = self.encoder(x.view(-1, H*W),code)
        z = self.sampling(mu, log_var)
        return self.decoder(z,code), mu, log_var


# Designing tip: make thin and wide DNN (not deep but wide)
class Proxy(nn.Module):
    def __init__(self,x_dim, h_dim):
        super(Proxy,self).__init__()

        self.fc1 = nn.Linear(x_dim,h_dim)
        self.fc2 = nn.Linear(h_dim,h_dim)
        self.fc3 = nn.Linear(h_dim,1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x





# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, H*W), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


# training for proxy model f(x)
def train_proxy(epoch,train_loader):
    proxy.train()
    train_loss = 0
    for batch_idx, (data, score) in enumerate(train_loader):
        data = data.cuda()
        score = score.cuda()
        optimizer_proxy.zero_grad()
        batch_size = data.shape[0]
        data = data.view(batch_size,1,-1)

        output = proxy(data)
        output = output.view(-1,1)

        loss = (output - score).pow(2).mean()
        
        loss.backward()
        train_loss += loss.item()
        optimizer_proxy.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    average_train_loss = train_loss / len(train_loader.dataset)

    return average_train_loss


# measure validation loss for proxy
def validation_proxy(epoch,validation_loader):
    proxy.eval()
    test_loss = 0
    for batch_idx, (data, score) in enumerate(validation_loader):
        data = data.cuda()
        score = score.cuda()
        
        batch_size = data.shape[0]
        data = data.view(batch_size,1,-1)

        output = proxy(data)
        # print(score)
        # print(output)
        # assert(False)
        loss = (output - score).pow(2).mean()
   
        test_loss += loss.item()
      
        
        if batch_idx % 100 == 0:
            print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(validation_loader.dataset),
                100. * batch_idx / len(validation_loader), loss.item() / len(data)))
    average_test_loss = test_loss / len(validation_loader.dataset)

    return average_test_loss


# train for generator
def train(epoch,train_loader):
    vae.train()
    train_loss = 0
    for batch_idx, (data, score) in enumerate(train_loader):
        data = data.cuda()
        score = score.cuda()
        optimizer_vae.zero_grad()
        batch_size = data.shape[0]
        data = data.view(batch_size,1,-1)

        recon_batch, mu, log_var = vae(data, score)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer_vae.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = PatternDataset(path ='data/version2/',train=True,transform=transform)
    dataset_validation = PatternDataset(path ='data/version2/',train=False,transform=transform)

    dataloader = DataLoader(dataset=dataset,
                        batch_size=64,
                        shuffle=True,
                        drop_last=False)   

    dataloader_validation = DataLoader(dataset=dataset_validation,
                        batch_size=64,
                        shuffle=True,
                        drop_last=False)   


    ##########################################################
    # build model
    ##########################################################

    # proxy model (score approximation of y = f(x))
    proxy = Proxy(x_dim = H*W, h_dim = 2048)

    # conditional VAE (generator for p(x|y))
    vae = VAE(x_dim=H*W, h_dim1= 1024, h_dim2=1024, z_dim=2)


    if torch.cuda.is_available():
        vae.cuda()
        proxy.cuda()


    # optimizer setting
    optimizer_proxy = optim.Adam(proxy.parameters(),lr=1e-4,
                            weight_decay=1e-6)
    optimizer_vae = optim.Adam(vae.parameters(),lr=1e-4)


    # training proxy model
    best_loss = 1e6
    early_stop_tol = 10
    for epoch in range(1000):
        _ = train_proxy(epoch,dataloader)
        test_loss = validation_proxy(epoch, dataloader_validation)

        # early stopping for preventing overfitting
        if best_loss > test_loss:
            best_loss = test_loss
            early_stop_count = 0
        else:
            early_stop_count +=1
        if early_stop_count >= early_stop_tol:
            print(best_loss)
            print('early stopping')
            break



    # training conditional generator
    for epoch in range(500):
        train(epoch,dataloader)



    # sampling from pretrained generator and proxy
    with torch.no_grad():

        # M=4000 for candidate sampling
        z = torch.randn(4000, 2).cuda()
        code = torch.ones(4000,1).cuda()
        sample = vae.decoder(z,code).cuda()

        # post-processing of image because we only have 0 or 1
        sample[sample>0.7]=1
        sample[sample<=0.7]=0

        proxy.eval()

        # psuedo scoring by pretrained proxy
        score = proxy(sample)
        sorted_indices = torch.argsort(score.view(-1), descending=True)


        # TopK screening using psuedo score
        k = 200
        topk_indices = sorted_indices[:k]
        topk_samples = sample[topk_indices,:]


        # save numpy data
        np_save = topk_samples.view(200, W, H).cpu().numpy()
        np.save('output.npy',np_save)

        # save image
        save_image(topk_samples.view(200, 1, W, H), './samples/sample_' + '.png')