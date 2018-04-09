from experimental.reinforce.discrete_main import *


def test_main():
    reward_filter = MeanStdFilter((), clip=5.)
    reward_filter2 = MeanStdFilter((), clip=5.)
    config = AttrDict(default_config())
    env = gym.make(config.env_name)
    env = ConvertTo32Bit(env)
    sess = tf.Session()
    policy = Policy(sess, config)
    poli = Policy(sess, config)
    
    traj = rollouts(env, policy, reward_filter, config)
    traj2 = rollouts(env, poli, reward_filter2, config)
    print(traj[0])
    print(traj2[0])


def test_plot_return():
    reward_filter = MeanStdFilter((), clip=5.)
    config = AttrDict(default_config())
    env = gym.make(config.env_name)
    env = ConvertTo32Bit(env)
    sess = tf.Session()
    policy = Policy(sess, config)
    
    traj = rollouts(env, policy, reward_filter, config)
    raw_returns = [t.raw_return for t in traj]
    returns = [t.return_ for t in traj]
    
    import matplotlib.pyplot as plt
    
    x = np.arange(len(traj))
    
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(x, raw_returns)
    axes[1].plot(x, returns)
    
    plt.show()


def plot_loss_score():
    config = AttrDict(default_config())
    # Define Agent that train with REINFORCE algorithm.
    agent = REINFORCE(config)
    
    # Train for num_iters times.
    episode_loss = []
    episode_score = []
    for i, losses in enumerate(agent.train(num_iters=100)):
        loss = np.mean(losses)
        # Evaluate the policy so that it will mean score.
        score = np.mean([evaluate_policy(agent.policy) for _ in range(10)])
        message = 'episode: {0}, loss: {1}, score: {2}'.format(i, loss, score)
        print('{0}{1}{2}'.format(bcolors.HEADER, message, bcolors.ENDC))
        episode_loss.append(loss)
        episode_score.append(score)
    x = np.arange(len(episode_loss))
    plt.plot(x, episode_loss)
    plt.plot(x, episode_score)
    plt.show()


def test_evaluate():
    sess = tf.Session()
    config = AttrDict(default_config())
    policy = Policy(sess, config)
    saver = tf.train.Saver()
    saver.restore(sess, './reinforce_deubgging')
    
    score = np.mean([evaluate_policy(policy, config) for _ in range(10)])
    print(score)


def main(_):
    test_evaluate()


if __name__ == '__main__':
    tf.app.run()