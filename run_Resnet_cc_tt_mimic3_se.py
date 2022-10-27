import os # system-wide functions
import sys
import numpy as np # For numerical computation
import warnings
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from vit_torch.ResNet_CC_TT import ResNet_CC_TT_SE
import scipy.io as sio

length = 1024
class BpDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        file = self.file_list[idx]
        sbp = int(file.split('_')[-3])
        dbp = int(file.split('_')[-2])
        raw_data = sio.loadmat(file)
        
        return raw_data['ppg_ac'][0][:length].reshape(1,-1), sbp, dbp

batch_size = 512

train_ = r'data/list_train.txt'
valid_ = r'data/list_valid.txt'

train_list = []
valid_list = []

with open(train_,'r') as fp:
    _=fp.readlines()
    for file in _:
        train_list.append(file.strip())

np.random.shuffle(train_list)
train_data = BpDataset(train_list)
train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )


with open(valid_,'r') as fp:
    _=fp.readlines()
    for file in _:
        valid_list.append(file.strip())

valid_data = BpDataset(valid_list)
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=False )

print(len(train_data), len(train_loader))
print(len(valid_data), len(valid_loader))


model = ResNet_CC_TT_SE(channels = 16, num_classes = 2).cuda()


# Training settings
epochs = 100
gamma = 0.7
lr = 5e-5
seed = 42

# loss function
criterion = nn.L1Loss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size = 20, gamma=gamma)

import random
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

log_path = 'log/mimic3_resnet18_channels_16_cc_tt_se'
if not os.path.exists(log_path):
    os.mkdir(log_path)
logger = SummaryWriter(log_dir=log_path)
model_path = 'model/mimic3_resnet18_channels_16_cc_tt_se'
if not os.path.exists(model_path):
    os.mkdir(model_path)

fw = open(log_path + '/' + log_path[4:] + '.txt','w')
try:
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_loss_sbp = 0
        epoch_loss_dbp = 0

        model.train()
        for data, sbp_target, dbp_target in tqdm(train_loader,disable=True):
            data = data.to("cuda").float()
            sbp_target = sbp_target.to("cuda")
            dbp_target = dbp_target.to("cuda")
            out = model(data)
            loss1 = criterion(out[:,0], sbp_target) 
            loss2 = criterion(out[:,1], dbp_target)
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sbp += loss1
            epoch_loss_dbp += loss2
        #scheduler.step()
        epoch_loss_sbp = epoch_loss_sbp / len(train_loader)
        epoch_loss_dbp = epoch_loss_dbp / len(train_loader)
        epoch_loss =  + epoch_loss_sbp + epoch_loss_dbp
        logger.add_scalar("learning rate", lr, global_step=epoch)
        logger.add_scalar("train_loss",epoch_loss,global_step=epoch)
        logger.add_scalar("train_loss_sbp",epoch_loss_sbp,global_step=epoch)
        logger.add_scalar("train_loss_dbp",epoch_loss_dbp,global_step=epoch)

        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0
            epoch_val_loss_sbp = 0
            epoch_val_loss_dbp = 0
            for data, sbp_target, dbp_target in valid_loader:
                data = data.to("cuda").float()
                sbp_target = sbp_target.to("cuda")
                dbp_target = dbp_target.to("cuda")
                out = model(data)
                val_loss1 = criterion(out[:,0], sbp_target) 
                val_loss2 = criterion(out[:,1], dbp_target)
                epoch_val_loss_sbp += val_loss1
                epoch_val_loss_dbp += val_loss2
            epoch_val_loss_sbp = epoch_val_loss_sbp / len(valid_loader)
            epoch_val_loss_dbp = epoch_val_loss_dbp / len(valid_loader)
            epoch_val_loss = epoch_val_loss_sbp + epoch_val_loss_dbp
            logger.add_scalar("valid_loss",epoch_val_loss,global_step=epoch)
            logger.add_scalar("valid_loss_sbp",epoch_val_loss_sbp,global_step=epoch)
            logger.add_scalar("valid_loss_dbp",epoch_val_loss_dbp,global_step=epoch)

        model_name  =   '%s/model_%d.pth' % (model_path, epoch)
        torch.save(model.state_dict(), model_name)
        cur_time    =   time.strftime("%H:%M:%S", time.localtime())
        print(
            f"{cur_time} Epoch : {epoch+1}/{epochs} - train_loss : {epoch_loss:.4f} - sbp : {epoch_loss_sbp:.4f} - dbp : {epoch_loss_dbp:.4f} - val_loss : {epoch_val_loss:.4f}- sbp : {epoch_val_loss_sbp:.4f}- dbp : {epoch_val_loss_dbp:.4f}"
        )

        fw.write(f"{cur_time} Epoch : {epoch+1}/{epochs} - train_loss : {epoch_loss:.4f} - sbp : {epoch_loss_sbp:.4f} - dbp : {epoch_loss_dbp:.4f} - val_loss : {epoch_val_loss:.4f}- sbp : {epoch_val_loss_sbp:.4f}- dbp : {epoch_val_loss_dbp:.4f}")
        fw.write('\n')
except KeyboardInterrupt:
    fw.close()
sys.exit()

