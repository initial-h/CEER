import gym
from collections import deque
import numpy as np

import arguments as args
from gym import spaces

class PreprocessWrapper(gym.Wrapper):
    def __init__(self, env, r_preprocess=None, s_preprocess=None):
        '''
        reward & state preprocess
        record info like real reward, episode length, etc
        Be careful: when an episode is done: check info['episode'] for information
        '''
        gym.Wrapper.__init__(self, env)
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
        state = self.env.reset()
        # state = state.astype('float32') # todo: can change to int8 on atari
        # preprocess state
        if self.s_preprocess is not None:
            state = self.s_preprocess(state)
        return state

    def render(self, mode='human'):
        return self.env.render(mode=mode)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None # set a fixed number
        self.noop_action = 0
        # assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            # noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
            noops = np.random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

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
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class BatchEnvWrapper:
    def __init__(self, envs,planning = False):
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
            if args.fix_difficulty:
                assert env.game.env.difficulty_ramp() == 0 or env.game.env.difficulty_ramp() == None,env.game.env.difficulty_ramp()
            # if reward !=0:
            #     print('!!!:',reward)
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
        env = gym.make(env_id)
        # env = FrameStack(env,2)
        env = TimeLimit(env, max_episode_steps=5000)
        # todo
        env.seed(i*1000) # random seed
        env = PreprocessWrapper(env)
        if args.deterministic:
            env.game.sticky_action_prob = 0.
            env = NoopResetEnv(env)
        if args.fix_difficulty:
            env.game.env.ramping = False
        envs.append(env)
    batch_env = BatchEnvWrapper(envs)
    return batch_env