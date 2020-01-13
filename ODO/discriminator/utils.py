#!/usr/local/bin/env python
import pandas as pd
import numpy as np
from random import random
import math
import os

from multiprocessing import Pool
from itertools import zip_longest

from rdkit import Chem
from rdkit.Chem import Descriptors

def get_headings(Ipc=False):
    headings = [desc[0] for desc in Descriptors.descList]
    if Ipc is False:
        headings.remove('Ipc')
    return headings

def prob_round(x):
    sign = np.sign(x)
    x = abs(x)
    is_up = random() < x-int(x)
    round_func = math.ceil if is_up else math.floor
    return sign * round_func(x)

def get_opt_input(data_dir, smi_file, vec_file, target_file, index=1):
    smi_file = data_dir+smi_file
    vec_file = data_dir+vec_file
    target_file = data_dir+target_file
    smi_data = pd.read_csv(smi_file, header=0).values
    vec_data = pd.read_csv(vec_file, header=0)
    vec_data = vec_data.reindex(columns=get_headings()).values
    target_data = pd.read_csv(target_file, header=0).values
    return vec_data[index], target_data[index], smi_data[index]

def get_float_bool(data_dir, file):
    file = data_dir+file
    float_bool = pd.read_csv(file, header=0, dtype=int)
    float_bool = float_bool.reindex(columns=get_headings())
    assert len(float_bool.values) == 1
    return float_bool.values[0]

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def calc_descrs_for_smiles(smi):
    failed = [smi]
    failed.extend([float('inf') for i in Descriptors.descList])

    if smi is None or smi == '':
        print('Attempted to pass empty smiles')
        return failed

    m = Chem.MolFromSmiles(smi)

    if m is None:
        print('RDkit could not pass smiles: {}'.format(smi))
        return failed

    try:
        discp = [y(m) for x,y in Descriptors.descList]
    except:
        print('RDkit could not pass smiles: {}'.format(smi))
        return failed

    '''
    Problems:
    nans not being caught
    WARNING: not removing hydrogen atom without neighbors
    if I remove smiles becasue RDkit is a waste of space then the targets are too long
    '''

    if np.isnan(discp).any():
        print('RDkit has returned a nan for smiles: {}'.format(smi))
        return failed

    res = [smi]
    res.extend(discp)
    return res

def get_latent_vecs(mols, data_dir, file_name, target=None, num_procs=4):
    headings = get_headings(Ipc=True)
    num_lines = int(min(50000, len(mols)))
    n = int(len(mols) / num_lines)

    if os.path.exists((data_dir+'input_mols_filtered.csv')):
        pass
    f1 = open(data_dir+'input_mols_filtered.csv', 'ab')
    file_name = data_dir + file_name
    f2 = open(file_name, 'ab')
    for i, group in enumerate(grouper(mols, num_lines)):
        if i == 0:
            smi_head = 'smiles'
            header = ','.join(headings)
        else:
            header = ''
            smi_head = ''
        print('Processing group {}/{}'.format(i, n))
        with Pool(processes=num_procs) as pool:
            res = pool.map(calc_descrs_for_smiles, group)

        found_inf = np.array([float('inf') in x for x in res])
        if found_inf.any():
            res = [x for x, y in zip(res, found_inf) if not y]
            smi = [x.pop(0) for x in res]
            np.savetxt(f1, smi, header=smi_head, fmt='%s', delimiter=',', comments='', newline='\n')
            np.savetxt(f2, res, header=header, fmt='%.18e', delimiter=',', comments='', newline='\n')
        else:
            smi = [x.pop(0) for x in res]
            np.savetxt(f1, smi, header=smi_head, fmt='%s', delimiter=',', comments='', newline='\n')
            np.savetxt(f2, res, header=header, fmt='%.18e', delimiter=',', comments='', newline='\n')

    f1.close()
    f2.close()
    if found_inf.any() and target is not None:
        if target is not None:
            f3 = open(data_dir + 'input_target_filtered.csv', 'ab')
            target = [x for x, y in zip(target, found_inf) if not y]
            np.savetxt(f3, target, header='Prop', fmt='%.18e', delimiter=',', comments='', newline='\n')
        raise ValueError('Found error in output of rdkit. A file called input_mols_filtered.csv has been writen to disk'
                         ' with problem SMILES removed, seguest using this as input_mols.csv')



