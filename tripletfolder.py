from torchvision import datasets
import os
import numpy as np
import random
import torch

class TripletFolder(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(TripletFolder, self).__init__(root + '/satellite', transform)

        # record the drone information
        drone_path = []
        drone_id = []
        drone_root = root + '/drone/'
        for folder_name in os.listdir(drone_root):
            folder_root = drone_root + folder_name
            if not os.path.isdir(folder_root):
                continue
            for file_name in os.listdir(folder_root):
                drone_path.append(folder_root + '/' + file_name)
                drone_id.append(int(folder_name))

        self.drone_path = drone_path
        self.drone_id = np.asarray(drone_id)

    def _get_pos_sample(self, target, index):
        pos_index = np.argwhere(self.drone_id == target)
        rand = np.random.permutation(len(pos_index))
        result_path = []
        for i in range(4):
           t = i%len(rand)
           tmp_index = pos_index[rand[t]][0]
           result_path.append(self.drone_path[tmp_index])
        return result_path

    def _get_neg_sample(self, target):
        neg_index = np.argwhere(self.drone_id != target)
        neg_index = neg_index.flatten()
        rand = random.randint(0,len(neg_index)-1)
        return self.drone_path[neg_index[rand]]

    def __getitem__(self, index):
        path, target = self.samples[index]
        original_target = int(os.path.basename(os.path.dirname(path)))
        # pos_path, neg_path
        pos_path = self._get_pos_sample(original_target, index)

        sample = self.loader(path)
        pos0 = self.loader(pos_path[0])
        pos1 = self.loader(pos_path[1])
        pos2 = self.loader(pos_path[2])
        pos3 = self.loader(pos_path[3])

        if self.transform is not None:
            sample = self.transform(sample)
            pos0 = self.transform(pos0)
            pos1 = self.transform(pos1)
            pos2 = self.transform(pos2)
            pos3 = self.transform(pos3)

        if self.target_transform is not None:
            target = self.target_transform(target)

        c,h,w = pos0.shape
        pos = torch.cat((pos0.view(1,c,h,w), pos1.view(1,c,h,w), pos2.view(1,c,h,w), pos3.view(1,c,h,w)), 0)
        pos_target = target
        return sample, target, pos, pos_target
