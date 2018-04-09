import itertools
import sys
from collections import defaultdict

import gym
import numpy as np
import tensorflow as tf

env = gym.make("CartPole-v0")

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("episodes", 5000, "iterations of episodes")
tf.flags.DEFINE_float("goal_reward", -20, "Mean reward which can be finished")
tf.flags.DEFINE_float("discounter", 0.99, "For ignore for far future")
tf.flags.DEFINE_float("learning_rate", 0.2, "Learning Rate")


def bins(clip_min, clip_max, interval=4):  # 4^4
    return np.linspace(clip_min, clip_max, interval + 1)[1:-1]


def digitize_state(obs):
    digitize = [np.digitize(s, bins=bins(env.observation_space.low[i],
                                         env.observation_space.high[i],
                                         interval=4))
        for i, s in enumerate(obs)]
    # 0 ~ 255
    return sum([x * (4 ** i) for i, x in enumerate(digitize)])


def make_epsilon_greedy_policy(Q, nA):
    # pi(a | s)
    def policy_fn(state, i_episode):
        eps = 0.5 * (0.99 ** i_episode)
        A = np.ones(nA, np.float64) * (eps / nA)
        best_action = np.argmax(Q[state])
        A[best_action] += 1 - eps
        return A  # pi(a | s)
    
    return policy_fn


Q = defaultdict(lambda: np.zeros(env.action_space.n))
policy = make_epsilon_greedy_policy(Q, env.action_space.n)
all_episode_reward = np.zeros(FLAGS.episodes)

for i_episode in range(FLAGS.episodes):
    if (i_episode + 1) % 100 == 0:
        print("\rEpisode {}/{}.".format(i_episode + 1, FLAGS.episodes), end="")
        sys.stdout.flush()
    
    obs = env.reset()
    state = digitize_state(obs)
    
    episode_reward = 0
    for t in itertools.count():
        
        action_prob = policy(state, i_episode)
        action = np.random.choice(np.arange(env.action_space.n), p=action_prob)
        
        next_obs, reward, done, info = env.step(action)
        next_state = digitize_state(next_obs)
        
        # There are no need to select next action,
        # Cause, q-learning is off-policy.
        
        # next_action_prob = policy(next_state, i_episode)
        # next_action = np.random.choice(np.arange(env.action_space.n),
        #                                p=next_action_prob)
        
        if done:
            reward = -200
        episode_reward += reward
        
        # Q-Learning
        td_target = reward + FLAGS.discounter * np.max(Q[next_state])
        td_delta = td_target - Q[state][action]
        Q[state][action] += FLAGS.learning_rate * td_delta
        
        if done:
            all_episode_reward = np.hstack((all_episode_reward[1:], [episode_reward]))
            break
        
        state = next_state

import pickle

file_name = "q_learning.pkl"
save_Q = {}

for key, val in Q.items():
    save_Q[key] = list(val)

with open(file_name, "wb") as f:
    pickle.dump(save_Q, f)