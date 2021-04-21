import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from model import AlexNet

import pdb

class Trainer:
    def __init__(self, in_channels, num_classes):
        self.model = AlexNet(in_channels, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.num_classes = num_classes

    def train(self, save_dir, train_loader, test_loader, num_epochs, learning_rate, device):
        # pdb.set_trace()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
        writer = SummaryWriter()
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(save_dir, train_loader, epoch, optimizer, device)
            test_loss, test_acc = self.validate(save_dir, test_loader, epoch, device)
            scheduler.step(test_acc)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.close()

    def train_epoch(self, save_dir, train_loader, epoch, optimizer, device):
        # pdb.set_trace()
        self.model.train()
        self.model.to(device)
        epoch_loss = 0
        correct = 0
        total = 0
        iter_num =0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            prediction = self.model(data)
            loss = self.ce(prediction, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(prediction.data, 1)
            correct += ((predicted == target).float()).sum().item()
            total += data.shape[0]
            iter_num +=1

        torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
        torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
        print("[epoch %d]: epoch loss = %f, lr= %f, acc = %f" % (epoch + 1, epoch_loss/iter_num, 
            optimizer.param_groups[0]['lr'], float(correct)/total))

        return epoch_loss/iter_num, float(correct)/total

    def validate(self, model_dir, test_loader, epoch, device):
        # pdb.set_trace()
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch+1) + ".model"))
            epoch_loss = 0
            correct = 0
            total = 0
            iter_num = 0
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                prediction = self.model(data)
                loss = self.ce(prediction, target)
                epoch_loss += loss.item()

                _, predicted = torch.max(prediction.data, 1)
                correct += ((predicted == target).float()).sum().item()
                total += data.shape[0]
                iter_num +=1

            print("[validate epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss/iter_num,
                                                               float(correct)/total))
        return epoch_loss/iter_num, float(correct)/total