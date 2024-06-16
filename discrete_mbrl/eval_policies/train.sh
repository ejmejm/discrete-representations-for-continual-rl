# Stochastic crossing env
python train_policy.py --env_name=minigrid-crossing-stochastic --goal_type=explore_right --n_envs=8 --train_steps=2_000_000
python train_policy.py --env_name=minigrid-crossing-stochastic --goal_type=goal --n_envs=8 --train_steps=2_000_000

# Stochastic door key env
python train_policy.py --env_name=minigrid-door-key-stochastic --goal_type=explore_right --n_envs=8 --train_steps=2_000_000
python train_policy.py --env_name=minigrid-door-key-stochastic --goal_type=goal --n_envs=8 --train_steps=2_000_000