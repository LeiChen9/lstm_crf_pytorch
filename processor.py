import numpy as np  
import pandas as pd 
import pdb, pickle, os
import torch 
from torch.utils.data import Dataset, DataLoader
from utils import *

def pad_tensor(vec, pad, dim):
    """
    Args:
        vec: tensor to pad
        pad: the size to pad to
        dim: dimension to pad
    Return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.tensor([0] * pad_size[dim], dtype=torch.long)], dim=dim).unsqueeze(-1)

class PadCollate:
    def __init__(self, dim=0):
        self.dim = dim  
    
    def pad_collate(self, batch):
        """
        Args:
            batch: list of (tensor, label)
        Return:
            xs: a tensor of all examples in "batch" after padding
            yx: a tensor of all labels in "batch" after padding
        """
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        batch = list(map(lambda x:
                            (pad_tensor(x[0], pad=max_len, dim=self.dim),
                            pad_tensor(x[1], pad=max_len, dim=self.dim)),
                            batch))
        xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        ys = torch.stack(list(map(lambda x: x[1], batch)), dim=0)
        return xs, ys 
    
    def __call__(self, batch):
        return self.pad_collate(batch)

class MyData(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path 
        self.train_path = os.path.join("../", data_path, 'train', 'train.csv')
        self.test_path = os.path.join("../", data_path, 'test', 'test.csv')
        self.training_data = self.data_extract()
        self.word2dic, self.tag2dic = self.get_word_tag_dic()

    def set_label(self, news, lb_lst, thing, label):
        if not thing == '0':
            idx = news.find(thing)
            for i in range(len(thing)):
                lb_lst[idx+i] = label
        return lb_lst

    def data_extract(self):
        df = pd.read_csv(self.train_path)
        df = df.fillna("0")
        res_lst = []
        self.lengths = []
        for idx, row in df.iterrows():
            txt_lst = [c for c in row['news']]
            self.lengths.append(len(txt_lst))
            label_lst = ['O' for c in row['news']]
            tri = row['trigger']
            obj = row['object']
            sub = row['subject']
            time = row['time']
            loc = row['location']
            label_lst = self.set_label(row['news'], label_lst, tri, 'TRI')
            label_lst = self.set_label(row['news'], label_lst, obj, 'OBJ')
            label_lst = self.set_label(row['news'], label_lst, sub, 'SUB')
            label_lst = self.set_label(row['news'], label_lst, time, 'TIM')
            label_lst = self.set_label(row['news'], label_lst, loc, 'LOC')
            res_lst.append((txt_lst, label_lst))
        res_lst = sorted(res_lst, key=by_len, reverse=True)
        return res_lst
    
    def get_word_tag_dic(self):
        word_to_ix = {'<PAD>': 0}
        tag_to_ix = {'PAD': 0}
        for sentence, tags in self.training_data:
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
            for tag in tags:
                if tag not in tag_to_ix:
                    tag_to_ix[tag] = len(tag_to_ix)
        word_to_ix[START_TAG] = len(word_to_ix)
        tag_to_ix[START_TAG] = len(tag_to_ix)
        word_to_ix[STOP_TAG] = len(word_to_ix)
        tag_to_ix[STOP_TAG] = len(tag_to_ix)
        return word_to_ix, tag_to_ix
    
    def prepare_sequence(self, seq):
        idxs = [self.word2dic[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)
    
    def prepare_tags(self, tags):
        idxs = [self.tag2dic[t] for t in tags]
        return torch.tensor(idxs, dtype=torch.long)
    
    def __getitem__(self, index):
        item = self.training_data[index]
        return self.prepare_sequence(item[0]), \
                self.prepare_tags(item[1])
    
    def __len__(self):
        return len(self.training_data)

if __name__ == "__main__":
    data = MyData('data')
    train_loader = DataLoader(data, batch_size=64, shuffle=False, collate_fn=PadCollate(dim=0))
    for batch_x, batch_y in train_loader:
        print(batch_x.shape)
        pdb.set_trace()