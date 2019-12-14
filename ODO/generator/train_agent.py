#!/usr/bin/env python

import torch
import pickle
import numpy as np
import time
import os
from shutil import copyfile
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from rdkit import Chem
from os import path


from .model import RNN
from .data_structs import Vocabulary, Experience, Dataset_gen
from .scoring_functions import get_scoring_function
from .utils import Variable, seq_to_smiles, fraction_valid_smiles, unique, get_latent_vector
from .vizard_logger import VizardLog

def train_agent(restore_prior_from='data/Prior.ckpt',
                restore_agent_from='data/Prior.ckpt',
                data=None,
                voc=None,
                scoring_function='arb_vec',
                scoring_function_kwargs={},
                save_dir=None, learning_rate=0.0005,
                batch_size=128, n_steps=3000,
                num_processes=0, sigma=60,
                experience_replay=0):

    start_time = time.time()

    loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True,
                        collate_fn=Dataset_gen.collate_fn)


    print('Loading prior and agent...')
    Prior = RNN(voc, len(data[0][1]))
    Agent = RNN(voc, len(data[0][1]))

    logger = VizardLog('data/logs')

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load('data/Prior.ckpt'))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        Prior.rnn.load_state_dict(torch.load('data/Prior.ckpt', map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), learning_rate)

    # Scoring_function
    scoring_function = get_scoring_function(scoring_function=scoring_function, num_processes=num_processes,
                                            **scoring_function_kwargs)

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)

    # Log some network weights that can be dynamically plotted with the Vizard bokeh app
    logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_ih")
    logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_hh")
    logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "init_weight_GRU_embedding")
    logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "init_weight_GRU_layer_2_b_ih")
    logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "init_weight_GRU_layer_2_b_hh")

    # Information for the logger
    step_score = [[], []]

    print("Model initialized, starting training...")

    for step, (_, vec_batch) in tqdm(enumerate(loader), total=len(loader)):
        real_vecs = vec_batch.float()

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size, real_vecs)

        # Get prior likelihood and score
        prior_likelihood, _ = Prior.likelihood(Variable(seqs), real_vecs)
        smiles = seq_to_smiles(seqs, voc)
        gen_vecs = get_latent_vector(smiles)
        data = [[x, y] for x, y in zip(gen_vecs, real_vecs)]
        score = scoring_function(data)

        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience)>4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4, real_vecs)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, score, prior_likelihood)
        experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
              step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")
        for i in range(math.ceil(batch_size/10)):
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                       prior_likelihood[i],
                                                                       augmented_likelihood[i],
                                                                       score[i],
                                                                       smiles[i]))
        # Need this for Vizard plotting
        step_score[0].append(step + 1)
        step_score[1].append(np.mean(score))

        # Log some weights
        logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "weight_GRU_layer_2_w_ih")
        logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "weight_GRU_layer_2_w_hh")
        logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "weight_GRU_embedding")
        logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "weight_GRU_layer_2_b_ih")
        logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "weight_GRU_layer_2_b_hh")
        logger.log("\n".join([smiles + "\t" + str(round(score, 2)) for smiles, score in zip \
                            (smiles[:12], score[:12])]), "SMILES", dtype="text", overwrite=True)
        logger.log(np.array(step_score), "Scores")

    # If the entire training finishes, we create a new folder where we save this python file
    # as well as some sampled sequences and the contents of the experience (which are the highest
    # scored sequences seen during training)
    if not save_dir:
        save_dir = 'data/results/run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    os.makedirs(save_dir)
    copyfile('train_agent.py', os.path.join(save_dir, "train_agent.py"))

    experience.print_memory(os.path.join(save_dir, "memory"))
    torch.save(Agent.rnn.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))

    seqs, agent_likelihood, entropy = Agent.sample(batch_size, real_vecs)
    prior_likelihood, _ = Prior.likelihood(Variable(seqs))
    prior_likelihood = prior_likelihood.data.cpu().numpy()
    smiles = seq_to_smiles(seqs, voc)
    score = scoring_function(smiles)
    with open(os.path.join(save_dir, "sampled"), 'w') as f:
        f.write("SMILES Score PriorLogP\n")
        for smiles, score, prior_likelihood in zip(smiles, score, prior_likelihood):
            f.write("{} {:5.2f} {:6.2f}\n".format(smiles, score, prior_likelihood))

if __name__ == "__main__":

    voc = Vocabulary(init_from_file="data/Voc")

    # Create a Dataset from a SMILES file
    if path.isfile('./data/vecs.dat'):
        print('Found vectors, reading from file...')
        data = Dataset_gen(voc, "data/mols.smi", vec_file='./data/vecs.dat')
    else:
        data = Dataset_gen(voc, "data/mols.smi", vec_file=None)

    mew = data.mew
    std = data.std

    train_agent(data=data, voc=voc, scoring_function_kwargs={'mew':mew, 'std':std})
