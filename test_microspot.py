from pathlib import Path
import pickle

import torch
from reward_wrapper import Go2
from rsl_rl.algorithms import PPO, PpoConfig
from rsl_rl.modules import ActorCritic, PolicyConfig

import genesis as gs


def get_last_ckpt_path(ckpt_folder: Path | str) -> Path | None:
    ckpt_folder = Path(ckpt_folder)

    last_index = None
    last_ckpt_path = None

    for ckpt_path in ckpt_folder.iterdir():
        if ckpt_path.is_dir():
            continue
        ckpt_parts = ckpt_path.stem.split("_")
        if len(ckpt_parts) != 2:
            continue
        index = int(ckpt_parts[-1])
        if last_index is None or last_index < index:
            index = last_index
            last_ckpt_path = ckpt_path
    return last_ckpt_path
        

def main(ckpt_path: Path | str, cfgs_path: Path | str, cpu: bool = False):
    ckpt = torch.load(ckpt_path)

    gs.init(backend=gs.cpu if cpu else gs.gpu)

    env_cfg, obs_cfg, reward_cfg, command_cfg = pickle.load(
        open(cfgs_path, 'rb')
    )
    env_cfg['resampling_time_s'] = -1

    env = Go2(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        eval=True,
        debug=True,
    )

    alg_cfg = PpoConfig()
    policy_cfg = PolicyConfig()
    device = torch.device('cuda:0')

    if env.num_privileged_obs is not None:
        num_critic_obs = env.num_privileged_obs 
    else:
        num_critic_obs = env.num_obs
    actor_critic = ActorCritic(env.num_obs,
                               num_critic_obs,
                               env.num_actions,
                               policy_cfg).to(device)
    alg = PPO(actor_critic, alg_cfg, device=device)
    actor_critic.load_state_dict(ckpt['model_state_dict'])
    alg.actor_critic.eval()
    alg.actor_critic.to(device)
    policy = alg.actor_critic.actor

    env.reset()
    env.commands[:, 0] = -1
    env.commands[:, 1] = 0
    env.commands[:, 2] = 0
    obs = env.get_observations()

    with torch.no_grad():
        stop = False
        n_frames = 0
        while not stop:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            n_frames += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cpu', action='store_true', default=False)
    # parser.add_argument('-v', '--vis', action='store_true', default=True)
    # parser.add_argument('-r', '--record', action='store_true', default=False)
    parser.add_argument('ckpt_folder', type=str)
    args = parser.parse_args()
    ckpt_folder = Path(args.ckpt_folder)

    ckpt_path = get_last_ckpt_path(ckpt_folder)
    cfgs_path = ckpt_folder / "cfgs.pkl"
    main(ckpt_path, cfgs_path, args.cpu)
