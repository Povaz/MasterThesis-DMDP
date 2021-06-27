# Master Thesis
## Delayed RL: A Belief Representation Approach

This is the project repository for the Master Thesis "Delayed Reinforcement Learning: A Belief Representation Approach",
written by Erick Venneri, supervised by Prof. Marcello Restelli and co-supervised Dott. Pierre Liotet.

### D-TRPO Algorithm
D-TRPO algorithm is accessible by running run_dtrpo.py script with the use_belief option, for example:
```
python run_dtrpo.py --use_belief --env=Pendulum-v0 --mode=train --seeds 0 --stochastic_delays --delay_proba=0.55 
--stoch_mdp_distrib=Gaussian --stoch_mdp_param=1.0 --epochs=2000 --steps_per_epoch=5000 --max_ep_len=250 --gamma=0.99
--delta=0.001 --v_hid=64 --v_l=1 --vf_lr=0.01 --v_iters=3 --pi_hid=64 --pi_l=2 --pretrain_epochs=2 
--epochs_belief_training=200 --pretrain_steps=10000 --size_pred_buf=100000 --batch_size_pred=10000 --train_enc_iters=4
--enc_l=1 --enc_dim=64 --enc_lr=0.01 --maf_lr=0.01 --enc_heads=2 --enc_causal --n_blocks_maf=5 --hidden_dim=8
--hidden_dim_maf=16 --save_period=100

python run_dtrpo.py --env=Pendulum-v0 --mode=test --seeds 0 --stochastic_delays --max_delay=50 --delay_proba=0.7
--epoch_load=2000 --test_episodes=50 --test_steps=250 --test_type=Test 
--save_dir ./output/dtrpo/Pendulum-Results/Results-Delay3/Pendulum-v0-21-01-05_03_16_864646
```
The rest of the arguments are documented in the script. D-TRPO results can be found in output/dtrpo folder.

### L2-TRPO Algorithm
L2-TRPO algorithm is accessible by running run_dtrpo.py script without the use_belief option:
```
python run_dtrpo.py --env=Pendulum-v0 --mode=train --seeds 0 1 2 --delay=5 --force_stoch_env --stoch_mdp_distrib=Triangular 
--stoch_mdp_param=1.0 --epochs=2000 --steps_per_epoch=5000 --max_ep_len=250 --delta=0.001 --v_hid=128 --v_l=1 --vf_lr=0.01 
--v_iters=3 --pi_hid=67 --pi_l=2 --pretrain_epochs=2 --pretrain_steps=10000 --size_pred_buf=100000 --batch_size_pred=10000
--train_enc_iters=1 --enc_lr=0.005 --enc_dim=68 --enc_heads=2 --enc_l=1 --enc_ff=8 --enc_causal --enc_pred_to_pi
--save_period=100

python run_dtrpo.py --env=Pendulum-v0 --mode=test --seeds 0 --delay=5 --test_episodes=50 --test_steps=250 --epoch_load=2000
--save_dir ./output/l2trpo/Pendulum-Results/Results-Delay5/Pendulum-v0-21-01-06_11_11_987028
```
The rest of the arguments are documented in the script. L2-TRPO results can be found in output/l2trpo folder.

### M-TRPO Algorithm
M-TRPO algorithm is accessible by running run_trpo.py script with the memoryless option:
```
python run_trpo.py --memoryless --mode=train --env=Pendulum-v0 --delay=3 --memoryless --seeds 0 1 2 3 4 --epochs=2000
--steps_per_epoch=5000 --max_ep_len=250 --gamma=0.99 --delta=0.001 --v_hid=64 --v_l=1 --vf_lr=0.01 --v_iters=3 
--pi_hid=64 --pi_l=2 --save_period=500

python run_trpo.py --memoryless --mode=test --env=Pendulum-v0 --delay=3 --seeds 0 --test_epoch=2000 --test_episodes=50
--test_steps=250 --save_dir ./output/trpo/Pendulum-Memoryless/Results-Delay3/Pendulum-v0-21-01-02_11_58_049693
```
The rest of the arguments are documented in the script. M-TRPO results can be found in output/trpo/Pendulum-Memoryless.

### A-TRPO Algorithm
A-TRPO algorithm is accessible by running run_trpo.py script without the memoryless option:
```
python run_trpo.py--mode=train --env=Pendulum-v0 --delay=3 --seeds 0 1 2 3 4 --epochs=2000 -steps_per_epoch=5000 
--max_ep_len=250 --gamma=0.99 --delta=0.001 --v_hid=64 --v_l=1 --vf_lr=0.01 --v_iters=3 --pi_hid=64 --pi_l=2 
--save_period=500

python run_trpo.py --mode=test --env=Pendulum-v0 --delay=3 --seeds 0 --test_epoch=2000 --test_episodes=50 --test_steps=250
--save_dir ./output/trpo/Pendulum-Augmented/Results-Delay3/Pendulum-v0-21-01-05_11_05_763071
```
The rest of the arguments are documented in the script. M-TRPO results can be found in output/trpo/Pendulum-Augmented.

### SARSA Algorithm
SARSA Algorithm is accessible by running run_sarsa.py script:
```
python run_sarsa.py --mode=train --env=Pendulum --delay=5 --seeds 0 1 2 --epochs=2000 --steps_per_epoch=5000
--max_ep_len=250 --lr=0.1 --s_space=15 --a_space=3

python run_sarsa.py --mode=test --env=Pendulum --delay=5 --seeds 0 --test_episode=50 --test_steps=250 --s_space=15
--a_space=3 --save_dir ./output/sarsa/Pendulum-Results/Results-Delay3\Pendulum-20-12-31_13_44_975089
```
The rest of the arguments are documented in the script. SARSA results can be found in output/sarsa.

### DSARSA Algorithm
DSARSA Algorithm is accessible by running run_sarsa.py script with the dsarsa option:
```
python run_sarsa.py --dsarsa --mode=train --env=Pendulum  --delay=5 --seed 5 6 7 8 9 --epochs=2000 --steps_per_epoch=5000
--max_ep_len=250 --lr=0.1

python run_sarsa.py --dsarsa --mode=test --env=Pendulum --delay=5 --seeds 0 --test_episode=50 --test_steps=250 --s_space=15
--a_space=3 --save_dir ./output/dsarsa/Pendulum-Results/Results-Delay3\Pendulum-21-01-02_12_48_641409
```
The rest of the arguments are documented in the script. DSARSA results can be found in output/dsarsa.

### Result visualization
Results and images shown in the Thesis can be found in the notebooks folder, specifically:
  - thesis_encoder notebook is dedicated to Encoder Parameter Tuning (Section 6.2) results;
  - thesis_det notebook is dedicated to Deterministic Enviroment (Section 6.3) results;
  - thesis_stoch notebook is dedicated to Stochastic Environment (Section 6.4) results.
  
