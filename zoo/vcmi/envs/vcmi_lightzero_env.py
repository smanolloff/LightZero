from ding.utils import ENV_REGISTRY
from ding.envs import BaseEnv, BaseEnvTimestep
from easydict import EasyDict
import gymnasium as gym
import copy
import numpy as np

from .vcmi_gym import VcmiEnv


@ENV_REGISTRY.register('vcmi_lightzero')
class VcmiEnvLightZero(BaseEnv):
    config = dict(
        env_id='VCMI',
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def _process_lz_obs(self, obs):
        self._lz_obs = {
            # 'observation': obs.flatten(),
            'observation': obs.swapaxes(0, 1).swapaxes(0, -1),
            # 'observation': obs,
            'action_mask': self._env.action_mask().astype('int8'),
            'to_play': -1
        }

        return self._lz_obs

    def __init__(self, cfg: EasyDict) -> None:
        self.cfg = cfg
        self._needs_init = True

    def reset(self) -> dict:
        self._eval_episode_return = 0
        if self._needs_init:
            self._needs_init = False
            self._env = VcmiEnv(
                "gym/generated/88/88-3stack-30K-01.vmap",
                encoding_type="float",
                reward_dmg_factor=5,
                step_reward_fixed=-100,
                step_reward_mult=1,
                term_reward_mult=0,
                reward_clip_tanh_army_frac=1,
                reward_army_value_ref=500,
                random_combat=1,
                max_steps=100,
            )

        obs, _ = self._env.reset()
        return self._process_lz_obs(obs)

    def step(self, action: int) -> BaseEnvTimestep:
        obs, rew, term, trunc, info = self._env.step(action)

        done = term or trunc
        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return

        return BaseEnvTimestep(self._process_lz_obs(obs), rew, done, info)

    def random_action(self):
        return np.random.choice(np.where(self._lz_obs["action_mask"] == 1)[0])

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        np.random.seed(seed)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._env.observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._env.action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=-500.0, high=500.0, shape=(1,), dtype=np.float32)

    def __repr__(self) -> str:
        return "LightZero VCMI Env"
