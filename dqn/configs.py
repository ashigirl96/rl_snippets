from dqn.networks import feed_forward_value

def default_config():
    conv_value_layers = [
        [32, 8, 4], [64, 4, 2], [64, 3, 1]]
    network = feed_forward_value
    learning_freq = 4
    learning_starts = 50000
    learning_rate = 0.001
    batch_size = 5
    gamma = 0.99
    amount = 4
    target_update_freq = 10_000
    grad_norm_clipping = 10
    stopping_crierion = None
    
    # Train episodes
    # episodes = 10000
    episodes = 1
    
    # Replay Buffer
    capacity = 500000
    frame_size = 4
    # replay_memory_init_size = 50000
    replay_memory_init_size = 500
    
    return locals()
