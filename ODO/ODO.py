#!/usr/bin/python

import torch
import torch.utils.data as Data
import numpy as np
from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
import pandas as pd

from discriminator.data_structs import Dataset_discrim
from discriminator.model import Net
from discriminator.optimize import Optimize
from discriminator.utils import get_headings, prob_round, get_opt_input

from generator.model import RNN
from generator.data_structs import Vocabulary
from generator.data_structs import Dataset_gen
from generator.utils import seq_to_smiles, get_latent_vector

class ODO(object):
    def __init__(self):
        self.opt_solution = ODO.discrim(self)
        print(self.opt_solution)

    def discrim(self, optimize=True):
        net = Net(n_feature=330, n_hidden=2000, n_output=1)     # define the network
        net.cuda()

        data = Dataset_discrim('./discriminator/data/input_train.csv')
        training_data, testing_data = torch.utils.data.random_split(data, [3000, 498])
        loader = Data.DataLoader(training_data, batch_size=250, shuffle=True,
                                 drop_last=True, collate_fn=Dataset_discrim.collate_fn)

        net.train()
        net.train_network(loader)
        net.save_ckpt()

        net.eval()
        net.get_r_squared(testing_data)

        #start vector should be passed in, or use a util to grab ('global' solutions)

        ###WAS HERE
        seed_vec = get_opt_input()

        if optimize:
            opt = Optimize(data, net, seed_vec, target=7.5)
            return [opt.property, opt.solution]
        else:
            return None

    def discrim_to_gen(self):
        int_bool = pd.read_cvs('./discriminator/data/int_bool.csv', heading=0)
        int_bool= int_bool.reindex(columns=get_headings()).values
        return int_bool

    def generate(self, load_weights='data/Prior.ckpt', batch_size=1):
        pass

if '__main__':
    ODO()





