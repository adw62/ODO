#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm
from os import path

from .data_structs import Dataset_gen, Vocabulary
from .model import RNN
from .utils import Variable, decrease_learning_rate
rdBase.DisableLog('rdApp.error')

def pretrain(data_dir, voc_file, vec_file, mol_file, save_to, restore_from=None):
    """Trains the Prior RNN"""
    voc_file = data_dir+voc_file
    vec_file = data_dir+vec_file
    mol_file = data_dir+mol_file
    save_to = data_dir+save_to

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file=voc_file)

    batch_size = 128

    # Create a Dataset from a SMILES file

    print('Found vectors, reading from file')
    data = Dataset_gen(voc, mol_file, vec_file=vec_file)

    loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True,
                        collate_fn=Dataset_gen.collate_fn)
    network_size = len(data[0][1])
    if network_size < 512:
        network_size = network_size*2
        print('Network expanded to size {}'.format(network_size))
    Prior = RNN(voc, network_size)

    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from))
    
    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr=0.001)
    for epoch in range(1):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        for step, (smi_batch, vec_batch) in tqdm(enumerate(loader), total=len(loader)):

            # Sample from DataLoader
            seqs = smi_batch.long()
            vecs = vec_batch.float()

            # Calculate loss
            log_p, _ = Prior.likelihood(seqs, vecs)
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data.item()))
                seqs, likelihood, _ = Prior.sample(batch_size, vecs)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")
                torch.save(Prior.rnn.state_dict(), save_to)

        # Save the Prior
        torch.save(Prior.rnn.state_dict(), save_to)
    return data.mew, data.std

if __name__ == "__main__":
    pretrain()
