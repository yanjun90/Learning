import torch
from train import Trainer
from dataloader import NIFTIData
import os
import argparse
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', default='T1_bet_2_0413.nii.gz')
parser.add_argument('--action', default="Training")

args = parser.parse_args()


features_dim = 141
bz = 4
lr = 0.001
num_epochs = 25
num_classes = 2
label_dict = {"health": 0, "patient": 1}
train_batch_size = 4
test_batch_size = 1

data_path = '../data'
model_dir = "./models"
results_dir = "./results"
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

train_kwargs = {'batch_size': train_batch_size}
test_kwargs = {'batch_size': train_batch_size}

trainer = Trainer(features_dim, num_classes)
if args.action == "Training":
    train_dataset = NIFTIData(root_path=data_path, label_dict=label_dict, mode="Training", data_file=args.data_file)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)

    val_dataset = NIFTIData(root_path=data_path, label_dict=label_dict, mode="Testing", data_file=args.data_file)
    val_loader = torch.utils.data.DataLoader(val_dataset,**train_kwargs)

    trainer.train(model_dir, train_loader, val_loader, num_epochs=num_epochs, learning_rate=lr, device=device)

if args.action == "Testing":
    trainer.test(model_dir, results_dir, num_epochs, actions_dict, device)
