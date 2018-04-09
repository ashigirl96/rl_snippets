import ray

ray.init(redirect_output=True)


@ray.remote
def f(x):
    x += 1
    return x


def main():
    object_id = f.remote(10)
    print(object_id)


if __name__ == '__main__':
    main()