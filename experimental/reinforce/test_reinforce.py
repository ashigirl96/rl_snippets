from experimental.reinforce.main import *


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


def main():
    test_plot_return()
    


if __name__ == '__main__':
    main()