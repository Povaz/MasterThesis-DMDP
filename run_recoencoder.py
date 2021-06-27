import json
import argparse
import matplotlib.pyplot as plt
from importlib import import_module
import numpy as np
import pickle
from datetime import datetime as dt
from datetime import timedelta
from utils.RECOEncoderCore import RandomPolicy
from utils.RECOEncoder import ReconstructionNetwork
from utils.TRPOSimulator_old import SinglePathSimulatorStoch
from utils.various import *
from torch.nn import MSELoss
from torch.nn import L1Loss
from torch.optim import Adam


def train(rec_network, simulator, optimizer, transformer_decoder=False, last_state=False):
    elapsed_time = timedelta(0)
    episode = 0
    network.train()
    reconstruction_losses = []
    visited_states = []
    state_dim = get_space_dim(env.state_space)

    if args.mae:
        loss = L1Loss(reduction='mean')
    else:
        loss = MSELoss(reduction='mean')

    while episode < args.train_episodes:
        start_time = dt.now()
        episode += 1

        # Samples new Trajectories and Unroll them
        samples = simulator.sample_trajectories()
        time_samples = dt.now() - start_time

        extended_states, mask, states = buffer_to_matrix(samples, state_dim, args.stochastic_delays)
        time_matrix = dt.now() - start_time
        predictions = rec_network(extended_states)
        time_pred = dt.now() - start_time
        pred_flat, states_flat = predictions[mask], states[mask]
        reconstruction_loss = loss(states_flat, pred_flat)
        time_loss = dt.now() - start_time
        reconstruction_losses.append(reconstruction_loss.detach().item())

        # Compute the Gradients and execute an optimizing step
        optimizer.zero_grad()
        reconstruction_loss.backward()
        optimizer.step()
        time_optim = dt.now() - start_time

        # Times
        time_optim -= time_loss
        time_loss -= time_pred
        time_pred -= time_matrix
        time_matrix -= time_samples
        time_samples = ''.join(str(time_samples).split('.')[0])
        time_matrix = ''.join(str(time_matrix).split('.')[0])
        time_pred = ''.join(str(time_pred).split('.')[0])
        time_loss = ''.join(str(time_loss).split('.')[0])
        time_optim = ''.join(str(time_optim).split('.')[0])


        # Print Episode Result
        elapsed_time += dt.now() - start_time
        elapsed_time_str = ''.join(str(elapsed_time).split('.')[0])
        update_message = '[EPISODE]: {0}\t[RECONSTRUCTION LOSS]: {1:.5f}\t[ELAPSED TIME]: {2} -> [SAMPLE: {3}, MATRIX: {4}, PRED: {5}, LOSS: {5}, OPTIM: {6}]'
        format_args = (episode, reconstruction_loss.detach(), elapsed_time_str, 
                time_samples, time_matrix, time_pred, time_loss)
        prGreen(update_message.format(*format_args))

        # Save Results: Reconstruction Loss Plot
        x = range(0, episode, 1)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Episodes')
        plt.ylabel('Reconstruction Loss')
        ax.scatter(x, reconstruction_losses)
        plt.savefig(args.save_dir + '/reco_losses.png')
        plt.close(fig)

        # Save Results: Reconstruction Loss Values
        ckpt = {
            'reconstruction_net_dict': rec_network.state_dict(),
            'reco_losses': reconstruction_losses
        }
        save_path = os.path.join(args.save_dir, 'encoder.pt')
        torch.save(ckpt, save_path)

        if args.save_states:
            # this doesnt work with the new implementation
            visited_states.append(states_flat)
            pickle.dump(visited_states, open(os.path.join(args.save_dir, 'visited_states.p'), "wb"))


def test(rec_network, simulator, env, last_state=False):
    # Policy/Value Function visualization
    episode = 0
    reco_mse = []
    reco_rmse = []
    reco_nmae = []
    state_dim = get_space_dim(env.state_space)
    rec_network.eval()


    # Test Loop
    while episode < args.test_episodes:
        # Reset Environment
        
        # Samples new Trajectories and Unroll them
        samples = simulator.sample_trajectories()

        extended_states, mask, states = buffer_to_matrix(samples, state_dim, args.stochastic_delays)
        predictions = rec_network(extended_states)
        pred_flat, states_flat = predictions[mask], states[mask]
        reco_loss = MSELoss(reduction='mean')(states_flat, pred_flat)

        # Compute normalized MAE over each State Dimension (and prepare print variables)
        mae_message = ''
        for i in range(state_dim):
            pred_i = pred_flat[:, i].detach()
            state_i = states_flat[:, i].detach()
            dim_norm_factor = env.state_space.high[i] - env.state_space.low[i]
            mae_i = L1Loss(reduction='mean')(pred_i, state_i) / dim_norm_factor
            mae_i_string = np.around(mae_i.tolist(), decimals=5)
            mae_message += '[nMAE ' + str(i+1) + '° DIM]: ' + str(mae_i_string) + '\t'
            reco_nmae.append(mae_i.item())

        # Print Episode Result
        if args.per_episode:
            update_message = '[EPISODE]: {0}\t[MSE LOSS]: {1:.5f}\t[RMSE LOSS]: {2:.5f}\t\t' + mae_message
            format_args = (episode, reco_loss.detach(), np.sqrt(reco_loss.detach()))
            prGreen(update_message.format(*format_args))

        # Save Episode Results
        reco_mse.append(reco_loss.detach())
        reco_rmse.append(np.sqrt(reco_loss.detach()))

        episode += 1

    # Final Results of the test:
    prYellow('# ----------------------------------------------------------'
             ' RESULTS '
             '---------------------------------------------------------- #')
    message = '| [AVG. MSE]: {0:.5f}\t[AVG. RMSE]: {1:.5f}\t'
    for i in range(env.state_space.shape[0]):
        index = np.arange(i, args.test_episodes*env.state_space.shape[0], env.state_space.shape[0])
        avg_nmae_i = np.around(np.average([reco_nmae[i] for i in index]), decimals=5)
        message += '[AVG. nMAE ' + str(i+1) + '° dim]: ' + str(avg_nmae_i) + '\t'
    message += ' |'
    avg_reco_mse = np.average(reco_mse)
    avg_reco_rmse = np.average(reco_rmse)
    format_args = (avg_reco_mse, avg_reco_rmse)
    prYellow(message.format(*format_args))
    prYellow('# --------------------------------------------------------------'
             '--------------------------------------------------------------- #')
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trust Region Policy Optimization (PyTorch)')

    # General Arguments for Training and Testing RECOEncoder
    parser.add_argument('--mode', default='test', type=str, choices=['train', 'test'])
    parser.add_argument('--env', default='PendulumDelayEnv', type=str, choices=['CartPoleDelayEnv',
                                                                                'MountainCarDelayEnv',
                                                                                'PendulumDelayEnv'])
    parser.add_argument('--train_episodes', default=200, type=int, help='Number of Episodes used for Training')
    parser.add_argument('--test_episodes', default=100, type=int, help='Number of Episodes used for Testing')
    parser.add_argument('--n_trajectories', default=50, type=int, help='Number of trajectories (Training)')
    parser.add_argument('--max_timesteps', default=250, type=int, help='Max timesteps of the trajectories (Simulator)')
    parser.add_argument('--delay', default=1, type=int, help='Timesteps of Delays (Environment)')
    parser.add_argument('--stochastic_delays', action='store_true', help='Use stochastic delays.')
    parser.add_argument('--seed', default=0, type=int, help='Seed for Result Reproducibility')
    parser.add_argument('--maxdelay', default=50, type=int, help='Maximum delay of the environment.')
    parser.add_argument('--adam_lr', default=1e-3, type=float, help='Adam Optimizer Learning Rate')
    parser.add_argument('--mae', action='store_true', help='Using MAE Loss instead of MSE Loss (default)')

    # Arguments for Model and Results save
    parser.add_argument('--save_dir', default='./output/recoencoder', type=str, help='Path to the directory containing the save.')
    parser.add_argument('--per_episode', action='store_true', help='Whether print Per Episode results in test or not')
    parser.add_argument('--save_states', action='store_true', help='Whether to save the List of Visited States during'
                                                                   'training in a Pickle or not (Space Required)')

    # Specific Arguments for RECOEncoder
    parser.add_argument('--encoder_ff_hid', default=16, type=int, help='Encoder FeedForward Layer Dimensions')
    parser.add_argument('--encoder_dim', default=16, type=int, help='Encoder Input Dimension')
    parser.add_argument('--n_head', default=4, type=int, help='Number of Heads (MultiAttention)')
    parser.add_argument('--encoder_layers', default=1, type=int, help='Number of Encoder Layers')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout of the Encoder network.')

    # RECOEncoder Options
    parser.add_argument('--rescale', help='Rescale the output of the network.', action='store_true')
    parser.add_argument('--append_pos_encoding', help='Append the positional encoding rather than summing.', action='store_true')
    parser.add_argument('--normlayers', help='Rescaling + NormLayers through the Network', action='store_true')
    parser.add_argument('--batchnorm', help='Rescaling + Replacing NormLayers with BatchNormLayers', action='store_true')
    parser.add_argument('--transformer_decoder', help='Rescaling + Replacing NormLayers with BatchNormLayers', action='store_true')
    parser.add_argument('--uniform', help='Training with a Uniform Distribution for Random Actions', action='store_true')
    parser.add_argument('--mask', help='Training with causal network using mask.', action='store_true')
    parser.add_argument('--last_state', action='store_true', help='Compute the Loss only over the Last State Prediction')

    # Inverted Pendulum Environment Option
    parser.add_argument('--pendulum_state', action='store_true', help='Only for InvertedPendulum. Training the Encoder'
                                                                      'retrieving the Angle from Cos and Sin instead of'
                                                                      'using directly Cos and Sin values')
    args = parser.parse_args()

    if args.save_states:
            raise NotImplementedError

    # Environment initialization
    load_model = None
    if args.mode == 'test':
        # Loading the Reconstruction Network
        args.save_model = next(filter(lambda x: '.pt' in x, os.listdir(args.save_dir)))
        load_model = os.path.join(args.save_dir, args.save_model)
        load_parameters = os.path.join(args.save_dir, 'model_parameters.txt')
        with open(load_parameters) as text_file:
            file_args = json.load(text_file)
        file_args = Bunch(file_args)
        file_args.mode = 'test'
        file_args.per_episode = args.per_episode
        file_args.test_episodes = args.test_episodes
        file_args.seed = args.seed
        file_args.uniform = args.uniform
        file_args.last_state = args.last_state
        args = file_args

    env = import_module('utils.module_env.' + args.env)
    simulator_env = getattr(env, args.env)
    try: 
        env = simulator_env(delay=args.delay, stochastic_delays=args.stochastic_delays, max_delay=args.maxdelay)
    except:
        raise NotImplementedError

    # Seeding for Reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Random Policy Initialization
    policy = RandomPolicy(args.n_trajectories, env.action_space.shape[0], uniform=args.uniform)

    # Build the network
    network = ReconstructionNetwork(env, args.encoder_dim,  args.n_head, args.encoder_ff_hid, args.encoder_layers,
                                    dropout=args.dropout, rescaling=args.rescale, normalized=args.normlayers,
                                    batchnorm=args.batchnorm, mask=args.mask,
                                    transformer_decoder=args.transformer_decoder,
                                    append_pos_encoding=args.append_pos_encoding, pendulum_state=args.pendulum_state)

    # ----- TRAIN MODE ----- #
    if args.mode == 'train':
        # Print the Network properties and save total number of Parameters
        args.network_parameters = network_print(network)

        # Instantiate the Optimizer
        optimizer = Adam(network.parameters(), lr=args.adam_lr)

        # Build a new Folder for the new trained network
        args.save_dir = get_output_folder(args.save_dir + '/' + args.env + '-Results', args.env)

        # Save the Parameters of this Run
        with open(args.save_dir + '/model_parameters.txt', 'w') as text_file:
            json.dump(args.__dict__, text_file, indent=2)

        # Construct the Simulator that will draw the trajectories
        # if args.stochastic_delays:
        simulator = SinglePathSimulatorStoch(simulator_env, policy, args.n_trajectories, args.max_timesteps,
                                    delay=args.delay, stochastic_delays=args.stochastic_delays, seed=args.seed)
        # else:
        #     simulator = SinglePathSimulator(simulator_env, policy, args.n_trajectories, args.max_timesteps,
        #                                 delay=args.delay, stochastic_delays=args.stochastic_delays, seed=args.seed)

        # Train Function
        train(network, simulator, optimizer, transformer_decoder=args.transformer_decoder, last_state=args.last_state)

    # ----- TEST MODE ----- #
    elif args.mode == 'test':
        # Print the Network properties
        network_print(network)

        # Fill the Network with Learnt Parameters
        ckpt = torch.load(load_model, map_location='cpu')
        network.load_state_dict(ckpt['reconstruction_net_dict'])

        # Test Function
        simulator = SinglePathSimulatorStoch(simulator_env, policy, 1, args.max_timesteps,
                                    delay=args.delay, stochastic_delays=args.stochastic_delays)
        test(network, simulator, env, last_state=args.last_state)
