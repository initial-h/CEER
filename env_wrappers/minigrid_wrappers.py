import gym
from gym import error, spaces, utils
# from env_wrappers.minigrid import gym_minigrid
import gym_minigrid
import cv2
import numpy as np
import random
from collections import deque
import arguments as args
# from env_wrappers.minigrid.gym_minigrid.wrappers import *
from gym_minigrid.wrappers import *

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

class MiniGridImageResize(gym.ObservationWrapper):

    def __init__(self, env, max_size=(40, 40)):
        super().__init__(env)
        self.max_size = max_size
        orig_size = self.observation_space.shape[:2]
        h, w = orig_size

        self.size = orig_size
        self.need_resize = False
        # if h > max_size[0] or w > max_size[0]:
        self.size = max_size
        self.need_resize = True
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.size + (3,), dtype=np.uint8)
        # print('MiniGridImageResize: orig_size={} size={}'.format(env.observation_space, self.observation_space))

    def observation(self, obs):
        if self.need_resize:
            obs = cv2.resize(obs,
                             self.size,
                             # interpolation=cv2.INTER_AREA
                             interpolation=cv2.INTER_NEAREST
                             )

            obs = np.asarray(obs, dtype=np.uint8)
        return obs

class MinigridActionCompress(gym.Wrapper):
    def __init__(self, env, env_name):
        super().__init__(env)
        # print(env_name.split('-', 1))
        domain, task = env_name.split('-', 1)
        if task.startswith('Empty') or task.startswith('FourRooms') \
            or task.startswith('LavaGap') \
            or task.startswith('SimpleCrossing') \
            or task.startswith('LavaCrossing') \
            or task.startswith('LavaCrossing') \
            or task.startswith('Dynamic-Obstacles'):
            self.action_space = gym.spaces.Discrete(3)

class EqualScaleMinigridStepPenaltyReward(gym.Wrapper):

    def __init__(self, env, test=False):
        super().__init__(env)
        self.test = test

    def step(self, a):
        obs, rew, done, info = self.env.step(a)
        orig_rew = rew
        # rew = self.env.unwrapped.max_steps if rew else -1
        rew = self.env.unwrapped.max_steps/100 if rew else -1/100

        # Bad early termination
        if done and orig_rew == 0 and self.env.unwrapped.step_count < self.env.unwrapped.max_steps:
            # rew = -self.env.unwrapped.max_steps
            rew = -self.env.unwrapped.max_steps/100

        return obs, rew, done, info

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            info["maxsteps_used"] = True
            done = True
        else:
            if done:
                info["maxsteps_used"] = False
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class FrameStack(gym.Wrapper):
    def __init__(self, env, nstack):
        gym.Wrapper.__init__(self, env)
        self.nstack = nstack
        self.wos = env.observation_space  # wrapped ob space
        self.stackedobs = [np.zeros(self.wos.shape, self.wos.dtype)]*self.nstack
        low = np.repeat(self.wos.low, self.nstack, axis=-1)
        high = np.repeat(self.wos.high, self.nstack, axis=-1)
        self.observation_space = spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def step(self,action):
        obs, rew, new, info = self.env.step(action)
        self.stackedobs.append(obs)
        self.stackedobs.pop(0)
        return np.concatenate(self.stackedobs,axis=-1), rew, new, info

    def reset(self):
        obs = self.env.reset()
        self.stackedobs = [np.zeros(self.wos.shape, self.wos.dtype)]*self.nstack
        self.stackedobs.append(obs)
        self.stackedobs.pop(0)
        # print('self.stackedobs :',self.stackedobs.dtype)
        return np.concatenate(self.stackedobs,axis=-1)

class PreprocessWrapper():
    def __init__(self, env, r_preprocess=None, s_preprocess=None):
        '''
        reward & state preprocess
        record info like real reward, episode length, etc
        Be careful: when an episode is done: check info['episode'] for information
        '''
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.r_preprocess = r_preprocess
        self.s_preprocess = s_preprocess
        # self.s_preprocess = lambda x:x/255.
        self.rewards = []

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # state = state.astype('float32') # todo: can change to int8 on atari
        self.rewards.append(reward)
        if done:
            # if no EpisodicLifeEnv_withInfos wrapper, update info here
            if not info.get('EpisodicLife'):
                # return None if there is no EpisodicLife
                eprew = sum(self.rewards)
                eplen = len(self.rewards)
                epinfo = {"r": round(eprew, 6), "l": eplen}
                assert isinstance(info,dict)
                if isinstance(info,dict):
                    info['episode'] = epinfo
                self.rewards = []
        # preprocess reward
        if self.r_preprocess is not None:
            reward = self.r_preprocess(reward)
        # preprocess state
        if self.s_preprocess is not None:
            state = self.s_preprocess(state)
        return state, reward, done, info

    def reset(self):
        # seed = np.random.randint(1,50)
        # self.env.seed(1)
        # self.env.seed(seed)
        state = self.env.reset()
        # state = state.astype('float32') # todo: can change to int8 on atari
        # preprocess state
        if self.s_preprocess is not None:
            state = self.s_preprocess(state)
        return state

    def render(self, mode='human'):
        return self.env.render(mode=mode)

def wrap_minigrid(env_id, im_size=(40, 40), test=False):
    env = gym.make(env_id)
    # env = RGBImgObsWrapper(env)
    if args.FullyObs_minigrid:
        env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env = EqualScaleMinigridStepPenaltyReward(env, test=test)
    # env = FrameStack(env,2)
    env = MinigridActionCompress(env, env_id)
    # env = MiniGridImageResize(env, im_size)
    env = TimeLimit(env, env.max_steps)

    return env

class BatchEnvWrapper:
    def __init__(self, envs):
        self.envs = envs
        self.observation_space = list(envs[0].observation_space.shape)
        # self.observation_space = [84,84,1]
        self.action_space = envs[0].action_space.n
        self.epinfobuf = deque(maxlen=100)

    def step(self, actions):
        states = []
        rewards = []
        dones = []
        infos = []
        for i, env in enumerate(self.envs):
            state, reward, done, info = env.step(actions[i])
            if done:
                info['terminal_state'] = state
                state = env.reset()
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            maybeepinfo = info.get('episode')

            if maybeepinfo:
                self.epinfobuf.append(maybeepinfo)

        # print(infos)
        # print('rewards:',rewards)
        return states, rewards, dones, infos

    def reset(self):
        return [self.envs[i].reset() for i in range(self.get_num_of_envs())]

    def render(self, mode='human'):
        return self.envs[0].render(mode=mode)

    def close(self):
        self.envs[0].close()

    def get_num_of_envs(self):
        return len(self.envs)

    def get_episode_rewmean(self):
        #print([epinfo['r'] for epinfo in self.epinfobuf])
        #input()
        return round(self.safemean([epinfo['r'] for epinfo in self.epinfobuf]),2)

    def get_episode_rewstd(self):
        #print([epinfo['r'] for epinfo in self.epinfobuf])
        return round(self.safestd([epinfo['r'] for epinfo in self.epinfobuf]),2)

    def get_episode_rewmax(self):
        return round(self.safemax([epinfo['r'] for epinfo in self.epinfobuf]),2)

    def get_list_of_episode(self):
        return [epinfo['r'] for epinfo in self.epinfobuf]

    def get_episode_lenmean(self):
        return round(self.safemean([epinfo['l'] for epinfo in self.epinfobuf]),2)

    def safemean(self,xs):
        return np.nan if len(xs) == 0 else np.mean(xs)

    def safemax(self, xs):
        return np.nan if len(xs) == 0 else np.max(xs)

    def safestd(self,xs):
        return np.nan if len(xs) == 0 else np.array(xs).std()

def Baselines_DummyVecEnv(env_id,num_env,array_obs=True):
    envs = []
    for i in range(num_env):
        env = wrap_minigrid(env_id)
        env = PreprocessWrapper(env)
        envs.append(env)
    batch_env = BatchEnvWrapper(envs)
    return batch_env