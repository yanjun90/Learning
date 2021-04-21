import os
import random
import torch
import numpy as np

from Read_nifti import read_nifti_file
import pdb

class NIFTIData(torch.utils.data.Dataset):
    def __init__(self, root_path, label_dict=None, mode="Training", data_file="T1_bet.nii.gz"):
        self.data_file = data_file
        if label_dict is None:
            label_dict = {"health": 0, "patient": 1}
        self.label_dict = label_dict
        data_path = os.path.join(root_path, mode)
        self.data_list = self.parse_nifti(data_path, data_file)
        random.shuffle(self.data_list)

    def parse_nifti(self, data_path, data_file):
        data_list = list()
        for label_name in self.label_dict.keys():
            sub_list = os.listdir(os.path.join(data_path, label_name))
            for sub in sub_list:
                if not os.path.isdir(os.path.join(data_path, label_name, sub)):
                    continue
                data_item_path = os.path.join(data_path, label_name, sub, self.data_file)
                data_item_label = self.label_dict[label_name]
                data_list.append((data_item_path, data_item_label))

        return data_list

    def __getitem__(self, index):
        data_item_path, data_item_label = self.data_list[index]
        data_item = read_nifti_file(data_item_path)
        data_item_ten = torch.from_numpy(data_item).float()

        return (data_item_ten, data_item_label)

    def debug_getitem(self, index=0):
        # pdb.set_trace()
        data_item_path, data_item_label = self.data_list[index]
        data_item = read_nifti_file(data_item_path)
        data_item_ten = torch.from_numpy(data_item).float()
        print(data_item_path, data_item.shape)

        return (data_item_ten, data_item_label)

    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":
    root_path = "./data"
    data_file = "T1_bet_2_0413.nii.gz"
    # data_file="T1_bet.nii.gz"
    dataset = NIFTIData(root_path, data_file=data_file)
    for i in range(len(dataset)):
        data, label = dataset.debug_getitem(index=i)
