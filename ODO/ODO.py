#!/usr/bin/python

import os

import torch
import torch.utils.data as Data
import numpy as np
from rdkit import Chem
from rdkit import rdBase
import pandas as pd
import math

from discriminator.data_structs import Dataset_discrim
from discriminator.model import Net
from discriminator.optimize import Optimize
from discriminator.utils import get_headings, prob_round, get_opt_input, get_float_bool, get_latent_vecs

from generator.model import RNN
from generator.data_structs import Vocabulary
from generator.data_structs import Dataset_gen
from generator.utils import seq_to_smiles, get_moments
from generator.train_prior import pretrain

rdBase.DisableLog('rdApp.error')

class ODO(object):
    def __init__(self):
        #Define file locations
        self.discrim_data_dir = './discriminator/data/'
        self.gen_data_dir = './generator/data/'
        self.gen_ckpt_file = self.gen_data_dir+'vector_based/Prior.ckpt'

        self.target = 9.2
        #Make a discriminative model and use finite differnces to solev this modle for a set of inputs predicted to give a set target
        self.y_property, self.x_solution = ODO.discrim(self, target_property=self.target)
        print('Optimized solution has activity {}'.format(self.y_property))

        #Convert the solution to the decriminative modle to a form that can be fed into a genereative model
        ODO.discrim_to_gen(self)

        mew = self.gen_data_dir + 'mew.dat'
        std = self.gen_data_dir + 'std.dat'
        moments = [mew, std]
        #train a generative model
        train_RNN = False
        if train_RNN:
            self.gen_ckpt_file = ODO.train_generator(self, moments)

        #Use a generative modle to produce smiles using output of discriminative modle as input
        print('Generating SMILES from proposed vectors using RNN weights at {}'.format(self.gen_ckpt_file))
        if not os.path.exists(mew) and os.path.exists(std):
            print('mew and std used to normalize generation input not found at {}, {}'.format(mew, std))
            moments = None
        generated_smis = ODO.generate(self, self.gen_data_dir, self.gen_ckpt_file, moments=moments)

        #convert smiles back into vectors to be tested by the discriminative model
        get_latent_vecs(generated_smis, self.discrim_data_dir, 'output_vecs.csv')
        try:
            generated_vecs = pd.read_csv(self.discrim_data_dir+'output_vecs.csv', header=0, dtype=np.float64).values
        except:
            raise ValueError('Try deleting {}'.format(self.discrim_data_dir+'output_vecs.csv'))

        predict = ODO.predict_with_discrim(self, generated_vecs)
        #print(predict)
        print(np.average(predict), np.std(predict))
        thresh = [True if x >= 6 else False for x in predict]
        print('{}/{}'.format(thresh.count(True), len(thresh)))

    def predict_with_discrim(self, vecs):
        data = Dataset_discrim(self.discrim_data_dir + 'input_train.csv', self.discrim_data_dir + 'input_target.csv')
        sx, sy = data.get_scaler()
        vecs = sx.transform(vecs)
        net = ODO.load_discrim(self, './net.ckpt', True)
        predict = [net(vec).cpu().detach().numpy() for vec in vecs]
        predict = sy.inverse_transform(predict)
        predict = [x[0] for x in predict]
        return predict

    def load_discrim(self, ckpt=False, eval=False):
        net = Net(n_feature=200, n_hidden=2000, n_output=1)  # define the network
        if ckpt:
            net.load_state_dict(torch.load(ckpt))
        if eval:
            net.eval()
        net.to('cuda')
        return net

    def discrim(self, target_property, optimize=True):
        net = ODO.load_discrim(self)

        if not os.path.exists(self.discrim_data_dir+'input_mols.csv'):
            raise ValueError('Please add mols to {}'.format(self.discrim_data_dir+'input_mols.csv'))

        if not os.path.exists(self.discrim_data_dir+'input_target.csv'):
            raise ValueError('Please add target data to {}'.format(self.discrim_data_dir+'input_target.csv'))

        if not os.path.exists(self.discrim_data_dir+'input_train.csv'):
            print('No training data found at {}'.format(self.discrim_data_dir+'input_train.csv'))
            print('Calculating vectors')
            mols = pd.read_csv(self.discrim_data_dir+'input_mols.csv', header=0)
            mols = [x[0] for x in mols.values]
            get_latent_vecs(mols, self.discrim_data_dir, 'input_train.csv')


        data = Dataset_discrim(self.discrim_data_dir+'input_train.csv', self.discrim_data_dir+'input_target.csv')
        frac_split = 0.7
        split = [math.floor(len(data)*frac_split), math.ceil(len(data)*(1-frac_split))]
        training_data, testing_data = torch.utils.data.random_split(data, split)
        loader = Data.DataLoader(training_data, batch_size=250, shuffle=True,
                                 drop_last=True, collate_fn=Dataset_discrim.collate_fn)

        net.train()
        net.train_network(loader)
        net.save_ckpt()

        net.eval()
        net.get_r_squared(testing_data)

        seed_vec, activity, smi_data = get_opt_input(self.discrim_data_dir, 'input_mols_filtered.csv',
                                                                    'input_train.csv', 'input_target.csv')
        print('Seed smiles {} and activity {}'.format(smi_data, activity))
        if optimize:
            opt = Optimize(data, net, seed_vec, target=target_property)
            msd = np.average((opt.solution-seed_vec)**2)
            print('MSD between seed and solution = {}'.format(msd))
            return opt.property, opt.solution
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

    def train_generator(self, moment_files):
        print('Training RNN')
        folder = 'vector_based/'
        if not os.path.exists(self.gen_data_dir+folder+'vecs.csv'):
            print('Gen Vectors....')
            mols = pd.read_csv(self.gen_data_dir+'mols.csv', header=0)
            mols = [x[0] for x in mols.values]
            get_latent_vecs(mols, self.gen_data_dir, folder+'vecs.csv')
        mew, std = pretrain(self.gen_data_dir, 'Voc', folder+'vecs.csv', 'input_mols_filtered.csv', folder+'Prior.ckpt')
        ckpt_file = self.gen_data_dir+folder+'Prior.ckpt'
        header = get_headings()
        np.savetxt(moment_files[0], np.array([mew]), header=','.join(header),
                   delimiter=',', comments='', newline='\n')
        np.savetxt(moment_files[1], np.array([std]), header=','.join(header),
                   delimiter=',', comments='', newline='\n')
        return ckpt_file

    def generate(self, data_dir, ckpt_file, mode='vectors', batch_size=1, samples=50, moments=None):
        modes = ['reinvent', 'vectors']
        network_size = 400
        if mode == 'reinvent':
            data = [np.zeroes(network_size)]
        elif mode == 'vectors':
            vec_file = self.gen_data_dir + 'vector_based/vecs.csv'
            if moments is None:
                #Calculate the mew and std used to normalize generation data
                data = pd.read_csv(vec_file, header=0)
                # correct heading order
                data = data.reindex(columns=get_headings())
                data = data.values
                mew, std = get_moments(data)
                # catch any zeros which will give nan when normalizing
                std = np.array([x if x != 0 else 1.0 for x in std])
            else:
                #read mew and std from file, this save some time and mem
                mew = pd.read_csv(moments[0], header=0).values
                std = pd.read_csv(moments[1], header=0).values

            vectors =  pd.read_csv(self.discrim_data_dir+'rounded.csv', header=0)
            vectors = vectors.reindex(columns=get_headings()).values
            vectors = (vectors - mew) / std

            #replace data with normalized vectors
            data = torch.FloatTensor(vectors)
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
        with open('./output_smi.csv', 'w') as file:
            for j, test_vec in enumerate(data):
                test_vec = test_vec.float()
                for i in range(samples):
                    seqs, prior_likelihood, entropy = Prior.sample(batch_size, test_vec)
                    smiles = seq_to_smiles(seqs, voc)[0]
                    if Chem.MolFromSmiles(smiles):
                        valid += 1
                        all_smi.append(smiles)
                        file.write(smiles + str(',{}\n'.format(j)))
        print("\n{:>4.1f}% valid SMILES".format(100 * (valid / (samples * len(data)))))
        return all_smi


if '__main__':
    ODO()





