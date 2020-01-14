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

from scipy import spatial

rdBase.DisableLog('rdApp.error')

class ODO(object):
    def __init__(self):
        #Define file locations
        self.discrim_data_dir = './discriminator/data/'
        self.gen_data_dir = './generator/data/'
        self.gen_ckpt_file = self.gen_data_dir+'vector_based/Prior.ckpt'

        self.target = 6.1
        self.mixing = 0.0
        #Make a discriminative model and use finite differnces to solev this modle for a set of inputs predicted to give a set target
        self.y_property, self.x_solution = ODO.discrim(self, target_property=self.target, train=False)

        print('Target activity {}, optimized solution achieved activity {}'.format(self.target, self.y_property[0]))

        #Convert the solution to the decriminative modle to a form that can be fed into a genereative model
        ODO.discrim_to_gen(self)

        mew = self.gen_data_dir + 'mew.dat'
        std = self.gen_data_dir + 'std.dat'
        moments = [mew, std]
        #train a generative model if needed
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
            generated_vecs = pd.read_csv(self.discrim_data_dir+'output_vecs.csv', header=0, dtype=np.float64)
            generated_vecs = generated_vecs.reindex(columns=get_headings()).values
        except:
            raise ValueError('Try deleting {}'.format(self.discrim_data_dir+'output_vecs.csv'))

        #test if generating smiles close to generation vectors
        #ODO.test_vector_msd(self, generated_vecs)

        predict = ODO.predict_with_discrim(self, generated_vecs)
        np.savetxt('./a_{}_m_{}.dat'.format(self.target, self.mixing), predict)
        print('Average activity {} and std {}'.format(np.average(predict), np.std(predict)))
        thresh_hold = 7.0
        thresh = [True if x >= thresh_hold else False for x in predict]
        smi_above_thresh = [[x, y] for x, y, z in zip(generated_smis, predict, thresh) if z is True]
        print('Number of compunds created = {}'.format(len(thresh)))
        print('Precent of compounds with activity above {} = {}'.format(thresh_hold,
                                                                        100*(thresh.count(True)/len(thresh))))
        for x in generated_smis:
            print(x)

    def test_vector_msd(self, generated_vectors):
        vectors = pd.read_csv(self.discrim_data_dir + 'rounded.csv', header=0)
        vectors = vectors.reindex(columns=get_headings()).values
        vec = vectors[0]
        msd = [np.average((vec-x)**2) for x in generated_vectors]
        print(msd)

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
        net = Net(n_feature=199, n_hidden=2000, n_output=1)  # define the network
        if ckpt:
            net.load_state_dict(torch.load(ckpt))
        if eval:
            net.eval()
        net.to('cuda')
        return net

    def discrim(self, target_property, train=True, optimize=True):
        if train:
            net = ODO.load_discrim(self)
        else:
            file = './net.ckpt'
            print('Loading discriminator from file {}'.format(file))
            net = ODO.load_discrim(self, file, True)

        if not os.path.exists(self.discrim_data_dir+'input_mols.csv'):
            raise ValueError('Please add mols to {}'.format(self.discrim_data_dir+'input_mols.csv'))

        if not os.path.exists(self.discrim_data_dir+'input_target.csv'):
            raise ValueError('Please add target data to {}'.format(self.discrim_data_dir+'input_target.csv'))

        if not os.path.exists(self.discrim_data_dir+'input_train.csv'):
            print('No training data found at {}'.format(self.discrim_data_dir+'input_train.csv'))
            print('Calculating vectors')
            mols = pd.read_csv(self.discrim_data_dir+'input_mols.csv', header=0)
            mols = [x[0] for x in mols.values]
            target = pd.read_csv(self.discrim_data_dir+'input_target.csv', header=0)
            target = [x[0] for x in target.values]
            get_latent_vecs(mols, self.discrim_data_dir, 'input_train.csv', target)


        data = Dataset_discrim(self.discrim_data_dir+'input_train.csv', self.discrim_data_dir+'input_target.csv')
        frac_split = 0.7
        train_len = math.floor(len(data)*frac_split)
        split = [train_len, len(data)-train_len]
        training_data, testing_data = torch.utils.data.random_split(data, split)
        loader = Data.DataLoader(training_data, batch_size=250, shuffle=True,
                                 drop_last=True, collate_fn=Dataset_discrim.collate_fn)

        if train:
            net.train()
            net.train_network(loader)
            net.save_ckpt()
            print('Testing discriminator...')
            net.eval()
            net.get_r_squared(testing_data)

        seed_vec, activity, smi_data = get_opt_input(self.discrim_data_dir, 'input_mols_filtered.csv',
                                                                    'input_train.csv', 'input_target.csv')
        pred_act = ODO.predict_with_discrim(self, [seed_vec])
        print('Seed smiles {} with true activity {} and predicted activity {}'.format(smi_data, activity, pred_act))
        if optimize:
            opt = Optimize(data, net, seed_vec, target=target_property)
            msd = np.average((opt.solution-seed_vec)**2)

            [neighbour_vec, neighbour_index] = ODO.k_near_search(self, opt.solution, self.gen_data_dir+'vector_based/vecs.csv')
            print(ODO.get_smiles_by_index(self, [neighbour_index]))
            msd_lib = np.average((opt.solution - neighbour_vec) ** 2)
            print('MSD between gen lib and og solution = {}'.format(msd_lib))

            mixing = self.mixing
            print('Mixing param has value {}'.format(mixing))
            opt.solution = neighbour_vec*mixing + opt.solution*(1-mixing)
            msd_lib = np.average((opt.solution-neighbour_vec)**2)
            print('MSD between gen lib and fudged solution = {}'.format(msd_lib))

            print('MSD between seed and solution = {}'.format(msd))

            if (neighbour_vec == seed_vec).all():
                print('Seed is neighbour')

            return opt.property, opt.solution
        else:
            return None

    def get_smiles_by_index(self, idxs):
        data = pd.read_csv(self.gen_data_dir+'input_mols_filtered.csv', header=0).values
        # correct heading order
        smiles = []
        for i in idxs:
            smiles.append(str(data[i][0]))
        return smiles

    def k_near_search(self, vector, lib_vec_file, num_neighbours=1):
        # vector is vector for compund we want to search for neigbours of
        # lib_vec_file is a libary of compunds in vector form which will be search for neighbours
        all_neigh_dist = []
        all_neigh_index = []
        all_neigh_vec = []
        chunksize = 100000
        print('Looking for nearest neighbour:')
        for i, chunk in enumerate(pd.read_csv(lib_vec_file, chunksize=chunksize, header=0)):
            print('Evaluating chunk {} of length {}'.format(i, len(chunk)))
            # correct heading order
            chunk = chunk.reindex(columns=get_headings())
            traning_data = chunk.values
            tree = spatial.KDTree(traning_data)
            ans = tree.query(np.array(vector), k=num_neighbours)
            all_neigh_dist.append(ans[0])
            all_neigh_index.append(ans[1] + (i * chunksize))
            all_neigh_vec.append(traning_data[ans[1]])

        result = [[x, y] for _, x, y in sorted(zip(all_neigh_dist, all_neigh_vec, all_neigh_index), key=lambda pair: pair[0])]
        return result[0]

    def discrim_to_gen(self):
        float_bool = get_float_bool(self.discrim_data_dir, 'float_bool.csv')
        all_rounded = []
        for i in range(50):
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
        network_size = 398
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
                #read mew and std from file, this save some time and memory
                mew = pd.read_csv(moments[0], header=0)
                std = pd.read_csv(moments[1], header=0)
                mew = mew.reindex(columns=get_headings()).values
                std = std.reindex(columns=get_headings()).values

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

        all_smi = set()
        valid = 0
        with open('./output_smi.csv', 'w') as file:
            for j, test_vec in enumerate(data):
                test_vec = test_vec.float()
                for i in range(samples):
                    seqs, prior_likelihood, entropy = Prior.sample(batch_size, test_vec)
                    smiles = seq_to_smiles(seqs, voc)[0]
                    if Chem.MolFromSmiles(smiles):
                        valid += 1
                        all_smi.add(smiles)
                        file.write(smiles + str(',{}\n'.format(j)))
        all_smi = list(all_smi)
        print("\n{:>4.1f}% valid SMILES".format(100 * (valid / (samples * len(data)))))
        return all_smi


if '__main__':
    ODO()





