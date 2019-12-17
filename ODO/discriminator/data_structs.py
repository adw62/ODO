#!/usr/local/bin/env python

import torch
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .utils import get_headings

class Dataset_discrim(Data.Dataset):
    def __init__(self, d_file, t_file):
        # Read and scale descriptors
        x = pd.read_csv(d_file, header=0)
        # Removes smiles colum
        x = x.reindex(columns=get_headings())
        x = x.values
        scalerx = MinMaxScaler(feature_range=(0, 1))
        scalerx.fit(x)
        x = scalerx.transform(x)
        self.scalerx = scalerx

        # Read and scale target property
        y = pd.read_csv(t_file, header=0)
        y = y.values
        scalery = MinMaxScaler(feature_range=(0, 1))
        scalery.fit(y)
        y = scalery.transform(y)
        self.scalery = scalery

        # Save data as pytorch arrays
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __getitem__(self, i):
        xi = self.x[i]
        yi = self.y[i]
        return ([Variable(xi), Variable(yi)])

    def __len__(self):
        assert len(self.x) == len(self.y)
        return len(self.x)

    def get_scaler(self):
        return [self.scalerx, self.scalery]

    @classmethod
    def collate_fn(cls, arr):
        """Function to take a list and turn them into a batch"""
        number_of_assays = 1
        max_length_x = max([data[0].size(0) for data in arr])
        max_length_y = number_of_assays
        # Got to set cuda here, in reinvent Variable is overloaded and this is implicit.
        collated_arr_x = Variable(torch.zeros(len(arr), max_length_x).cuda())
        collated_arr_y = Variable(torch.zeros(len(arr), max_length_y).cuda())
        for i, data in enumerate(arr):
            collated_arr_x[i, :data[0].size(0)] = data[0]
            collated_arr_y[i, :number_of_assays] = data[1]
        return collated_arr_x, collated_arr_y
