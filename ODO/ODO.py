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
from discriminator.utils import get_headings, prob_round, get_opt_input, get_float_bool

from generator.model import RNN
from generator.data_structs import Vocabulary
from generator.data_structs import Dataset_gen
from generator.utils import seq_to_smiles, get_latent_vector
from generator.train_prior import pretrain

class ODO(object):
    def __init__(self):
        self.discrim_data_dir = './discriminator/data/'
        self.gen_data_dir = './generator/data/'
        self.y_property, self.x_solution = ODO.discrim(self)
        print('Optimized solution has activity {}'.format(self.y_property))
        #print(self.opt_solution)
        ODO.discrim_to_gen(self)
        #RNN traning could be done here should return ckpt file path
        print('Generating SMILES from preposed vectors')
        ODO.generate(self, self.gen_data_dir, 'vector_based/Prior.ckpt')

    def discrim(self, optimize=True):
        net = Net(n_feature=330, n_hidden=2000, n_output=1)     # define the network
        net.cuda()

        data = Dataset_discrim(self.discrim_data_dir+'input_train.csv')
        training_data, testing_data = torch.utils.data.random_split(data, [3000, 498])
        loader = Data.DataLoader(training_data, batch_size=250, shuffle=True,
                                 drop_last=True, collate_fn=Dataset_discrim.collate_fn)

        net.train()
        net.train_network(loader)
        net.save_ckpt()

        net.eval()
        net.get_r_squared(testing_data)

        self.seed_vec, self.activity, self.smi_data = get_opt_input(self.discrim_data_dir, 'mols.smi', 'input_train.csv')
        print('Seed smiles {} and activity {}'.format(self.smi_data, self.activity))
        if optimize:
            opt = Optimize(data, net, self.seed_vec, target=7.5)
            return [opt.property, opt.solution]
        else:
            return None

    def discrim_to_gen(self):
        float_bool = get_float_bool(self.discrim_data_dir, 'float_bool.csv')
        all_rounded = []
        for i in range(5):
            rounded = []
            for x, is_float in zip(self.x_solution, float_bool):
                if is_float == 1:
                    rounded.append(x)
                else:
                    rounded.append(prob_round(x))
            if rounded not in all_rounded:
                all_rounded.append(rounded)
        all_rounded = np.array(all_rounded)
        header = get_headings()
        np.savetxt(self.discrim_data_dir+'rounded.csv', all_rounded, header=','.join(header),
                   delimiter=',', comments='', newline='\n')

    def train_generator(self, mode, ckpt_dir='./generator/agent'):
        modes = ['reinvent', 'vectors']
        job = ['train_prior', 'train_vector']
        return ckpt_file

    def generate(self, data_dir, ckpt_file, mode='vectors', batch_size=1, samples=50):
        ckpt_file = data_dir + ckpt_file
        modes = ['reinvent', 'vectors']
        if mode == 'reinvent':
            network_size = 330
            data = [np.zeroes(network_size)]
        elif mode == 'vectors':
            vec_file = self.gen_data_dir + 'vector_based/vecs.csv'
            data = pd.read_csv(self.discrim_data_dir + 'rounded.csv')
            data = data.reindex(columns=get_headings())
            data = data.values
            _, mew, std = get_latent_vector(None, vec_file, moments=True)
            data = [(x - mew) / std for x in data]
            data = torch.FloatTensor(data)
            network_size = len(data[0])
            #should pad here if network is too small < 300
        else:
            raise ValueError('Supported generation modes are {}'.format(modes))

        voc = Vocabulary(init_from_file=self.gen_data_dir+'Voc')
        Prior = RNN(voc, network_size)

        if torch.cuda.is_available():
            Prior.rnn.load_state_dict(torch.load(ckpt_file))
        else:
            Prior.rnn.load_state_dict(torch.load(ckpt_file, map_location=lambda storage, loc: storage))

        all_smi = []
        valid = 0
        for j, test_vec in enumerate(data):
            #print('Test vector {}'.format(test_vec))
            test_vec = test_vec.float()
            for i in range(samples):
                seqs, prior_likelihood, entropy = Prior.sample(batch_size, test_vec)
                smiles = seq_to_smiles(seqs, voc)[0]
                if Chem.MolFromSmiles(smiles):
                    valid += 1
                    all_smi.append(smiles + str(',{}'.format(j)))

        with open('./output.smi', 'w') as file:
            for smi in all_smi:
                file.write('{}\n'.format(smi))
        print("\n{:>4.1f}% valid SMILES".format(100 * (valid / (samples * len(data)))))

if '__main__':
    ODO()





