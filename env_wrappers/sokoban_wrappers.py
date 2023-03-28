import gym
import cv2
import numpy as np
import random
from collections import deque
from gym.wrappers import TimeLimit
# from env_wrappers.sokoban.gym_sokoban.envs.sokoban_env import SokobanEnv
from gym_sokoban.envs.sokoban_env import SokobanEnv
from gym_sokoban.envs.room_utils import generate_room

class SokobanEnv_new(SokobanEnv):
    def reset(self, second_player=False, render_mode='tiny_rgb_array'):
        try:
            self.room_fixed, self.room_state, self.box_mapping = generate_room(
                dim=self.dim_room,
                num_steps=self.num_gen_steps,
                num_boxes=self.num_boxes,
                second_player=second_player
            )
        except (RuntimeError, RuntimeWarning) as e:
            print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
            print("[SOKOBAN] Retry . . .")
            return self.reset(second_player=second_player)

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = self.render(render_mode)
        return starting_observation


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

class SokobanWrapper(gym.Wrapper):

    def __init__(self, env, screen_size=(84, 84)):
        super().__init__(env)
        self.screen_size = screen_size
        self.action_space = gym.spaces.Discrete(self.env.action_space.n - 1)
        # self.observation_space = gym.spaces.Box(low=0, high=255,
        #                                         shape=self.screen_size + (3,), dtype=np.uint8)
        screen_height, screen_width = (env.dim_room[0], env.dim_room[1])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)

    def _process_obs(self, obs):
        transformed_image = cv2.resize(obs,
                                       self.screen_size,
                                       interpolation=cv2.INTER_NEAREST)
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return int_image

    def reset(self):
        image = self.env.reset(render_mode='tiny_rgb_array')
        # processed_image = self._process_obs(image)
        # return processed_image
        return image

    def step(self, action):
        action = action + 1 # To remove noop
        observation, reward, is_terminal, info = self.env.step(action, 'tiny_rgb_array')
        # print('Max steps?', self.environment._check_if_maxsteps(), self.environment.max_steps, self.environment.num_env_steps, is_terminal)
        game_over = info.get("all_boxes_on_target", False) # Only send termination when all boxes are on the targets
        # processed_image = self._process_obs(observation)
        # return processed_image, reward, is_terminal, info
        return observation, reward, game_over, info

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
        seed = np.random.randint(1,50)
        # setup_seed(1) # 1, 17, 38
        setup_seed(seed)
        state = self.env.reset()
        # state = state.astype('float32') # todo: can change to int8 on atari
        # preprocess state
        if self.s_preprocess is not None:
            state = self.s_preprocess(state)
        return state

    def render(self, mode='human'):
        return self.env.render(mode=mode)


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
    size_strs, num_boxes, max_steps = env_id.split('_')[1:]
    sizes = tuple(map(lambda size_str: int(size_str), size_strs.split('x')))
    max_steps = int(max_steps)
    # print(sizes,num_boxes,max_steps)
    for i in range(num_env):
        if env_id.startswith('Sokoban-Push'):
            # print('Use PushOnly Sokoban, max_steps={}'.format(max_steps))
            env = SokobanWrapper(SokobanEnv_new(dim_room=sizes, num_boxes=int(num_boxes), max_steps=max_steps))
        else:
            raise NotImplemented()
        env = TimeLimit(env,env.max_steps)
        env = PreprocessWrapper(env)
        envs.append(env)
    batch_env = BatchEnvWrapper(envs)
    return batch_env