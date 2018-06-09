def default_config():
    conv_value_layers = [
        [32, 8, 4], [64, 4, 2], [64, 3, 1]]
    learning_freq = 4
    learning_starts = 50000
    learning_rate = 0.001
    batch_size = 5
    gamma = 0.99
    grad_norm_clipping = 10
    stopping_crierion = None
    
    update_target_estimator_every = 10000
    
    # Train episodes
    # episodes = 10000
    num_episodes = 1
    
    # Replay Buffer
    capacity = 500_000
    frame_size = 4
    replay_memory_init_size = 50000
    
    # eps-greedy policy coefficient
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay_steps = 500_000
    
    # max step size
    max_total_step_size = 1_000_000
    
    return locals()