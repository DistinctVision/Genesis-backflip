import typing as tp
import argparse
import os
import pickle
import shutil
from dataclasses import dataclass, field

import numpy as np
import torch
import wandb
from reward_wrapper import Go2
from locomotion_env import LocoEnv, ObservationConfig
from rsl_rl.modules import PolicyConfig
from rsl_rl.algorithms import PpoConfig
from rsl_rl.runners import OnPolicyRunner, TrainConfig

import genesis as gs


def get_cfgs():
    env_cfg = {
        'urdf_path': 'urdf/spotmicro.urdf',
        'links_to_keep': [],
        'num_actions': 12,
        'num_dofs': 12,
        # joint/link names
        'default_joint_angles': {  # [rad]
            'front_left_shoulder': 0.0,
            'front_left_leg': 0.0,
            'front_left_foot': 0.0,
            'front_right_shoulder': 0.0,
            'front_right_leg': 0.0,
            'front_right_foot': 0.0,
            'rear_left_shoulder': 0.0,
            'rear_left_leg': 0.0,
            'rear_left_foot': 0.0,
            'rear_right_shoulder': 0.0,
            'rear_right_leg': 0.0,
            'rear_right_foot': 0.0,
        },
        'dof_names': [
            'front_left_shoulder',
            'front_left_leg',
            'front_left_foot',
            'front_right_shoulder',
            'front_right_leg',
            'front_right_foot',
            'rear_left_shoulder',
            'rear_left_leg',
            'rear_left_foot',
            'rear_right_shoulder',
            'rear_right_leg',
            'rear_right_foot',
        ],
        'termination_contact_link_names': ['base_link'],
        'penalized_contact_link_names': ['base_link', 'rear_link', 'lidar_link'],
        'feet_link_names': ['foot'],
        'base_link_name': ['base'],
        # PD
        'PD_stiffness': {'shoulder': 30.0, 'leg': 30.0, 'foot': 30.0},
        'PD_damping': {'shoulder': 1.5, 'leg': 1.5, 'foot': 1.5},
        'use_implicit_controller': False,
        # termination
        'termination_if_roll_greater_than': 0.4,
        'termination_if_pitch_greater_than': 0.4,
        'termination_if_height_lower_than': 0.0,
        # base pose
        'base_init_pos': [0.0, 0.0, 0.42],
        'base_init_quat': [1.0, 0.0, 0.0, 0.0],
        # random push
        'push_interval_s': -1,
        'max_push_vel_xy': 1.0,
        # time (second)
        'episode_length_s': 20.0,
        'resampling_time_s': 4.0,
        'command_type': 'ang_vel_yaw',  # 'ang_vel_yaw' or 'heading'
        'action_scale': 0.25,
        'action_latency': 0.02,
        'clip_actions': 100.0,
        'send_timeouts': True,
        'control_freq': 50,
        'decimation': 4,
        'feet_geom_offset': 1,
        'use_terrain': False,
        # domain randomization
        'randomize_friction': True,
        'friction_range': [0.2, 1.5],
        'randomize_base_mass': True,
        'added_mass_range': [-1., 3.],
        'randomize_com_displacement': True,
        'com_displacement_range': [-0.01, 0.01],
        'randomize_motor_strength': False,
        'motor_strength_range': [0.9, 1.1],
        'randomize_motor_offset': True,
        'motor_offset_range': [-0.02, 0.02],
        'randomize_kp_scale': True,
        'kp_scale_range': [0.8, 1.2],
        'randomize_kd_scale': True,
        'kd_scale_range': [0.8, 1.2],
        # coupling
        'coupling': False,
    }
    reward_cfg = {
        'tracking_sigma': 0.25,
        'soft_dof_pos_limit': 0.9,
        'base_height_target': 0.3,
        'reward_scales': {
            'tracking_lin_vel': 1.0,
            'tracking_ang_vel': 0.5,
            'lin_vel_z': -2.0,
            'ang_vel_xy': -0.05,
            'orientation': -10.,
            'base_height': -50.,
            'torques': -0.0002,
            'collision': -1.,
            'dof_vel': -0.,
            'dof_acc': -2.5e-7,
            'feet_air_time': 1.0,
            'collision': -1.,
            'action_rate': -0.01,
        },
    }
    command_cfg = {
        'num_commands': 4,
        'lin_vel_x_range': [-1.0, 1.0],
        'lin_vel_y_range': [-1.0, 1.0],
        'ang_vel_range': [-1.0, 1.0],
    }

    return env_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', type=str, default='Go2')
    parser.add_argument('-v', '--vis', action='store_true', default=False)
    parser.add_argument('-c', '--cpu', action='store_true', default=False)
    parser.add_argument('-B', '--num_envs', type=int, default=8000)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('-o', '--online', action='store_true', default=False)

    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--ckpt', type=int, default=1000)
    args = parser.parse_args()

    if args.debug:
        args.vis = True
        args.offline = True
        args.num_envs = 1

    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level='warning',
    )

    log_dir = f'logs/{args.exp_name}'
    env_cfg, reward_cfg, command_cfg = get_cfgs()

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    obs_cfg = ObservationConfig()
    obs_cfg.num_obs = 9 + 3 * env_cfg['num_dofs']
    obs_cfg.num_history_obs = 1
    obs_cfg.num_privileged_obs = 12 + 4 * env_cfg['num_dofs']

    env = Go2(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
        eval=args.eval,
        debug=args.debug,
    )

    train_cfg = TrainConfig()
    train_cfg.experiment_name = args.exp_name
    train_cfg.max_iterations = args.max_iterations

    alg_cfg = PpoConfig()
    policy_cfg = PolicyConfig()

    runner = OnPolicyRunner(env, train_cfg, alg_cfg, policy_cfg, log_dir, device='cuda:0')

    if args.resume is not None:
        resume_dir = f'logs/{args.resume}'
        resume_path = os.path.join(resume_dir, f'model_{args.ckpt}.pt')
        print('==> resume training from', resume_path)
        runner.load(resume_path)

    wandb.init(project='genesis', name=args.exp_name, dir=log_dir, mode='online' if args.online else 'offline')

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg],
        open(f'{log_dir}/cfgs.pkl', 'wb'),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    main()


'''
# training
python train_backflip.py -e EXP_NAME

# evaluation
python eval_backflip.py -e EXP_NAME --ckpt NUM_CKPT
'''