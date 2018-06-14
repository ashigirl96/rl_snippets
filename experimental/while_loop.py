import numpy as np

F = lambda x: x


def main():
    H = 3
    batch_size = 32
    s = np.random.uniform(size=(batch_size, H))
    s_estimate = s[:, 0]
    print(s_estimate)
    for h in range(H):
        s_estimate = s_estimate + F(s_estimate)
        print(s_estimate)


if __name__ == '__main__':
    main()