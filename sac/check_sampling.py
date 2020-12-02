from datetime import datetime
import random
import numpy as np
import tqdm
import matplotlib.pyplot as plt


def measure_random_choices_time(buffer_size, sample_size):
    weights = np.random.randn(buffer_size)
    weights = (weights - min(weights)) / (max(weights) - min(weights))
    np_weights = np.asarray([1/buffer_size] * buffer_size)
    start = datetime.now()
    batch = random.choices(range(buffer_size), k=sample_size, weights=weights)
    # batch = random.choices(range(buffer_size), k=sample_size)
    end = datetime.now()
    start2 = datetime.now()
    batch = np.random.choice(range(buffer_size), size=sample_size, p=np_weights)
    # batch = np.random.choice(range(buffer_size), size=sample_size)
    end2 = datetime.now()
    spend_ms = (end - start).microseconds / 1000
    spend_ms_np = (end2 - start2).microseconds / 1000

    return spend_ms, spend_ms_np


if __name__ == '__main__':
    buffer_sizes = [1000000]
    # bs = 1000000
    sample_size = 256  # [2**i for i in range(1, 13)]
    time = []
    np_time = []

    for bs in buffer_sizes:
        random_time, np_random_time = measure_random_choices_time(bs, sample_size)
        time.append(random_time)
        np_time.append(np_random_time)
    print(len(buffer_sizes))
    print(np.mean(time))
    print(np.mean(np_time))

    plt.plot(buffer_sizes, time, buffer_sizes, np_time)
    plt.legend(['random.choices', 'np random choice '])
    plt.title('random.choice vs np.random.choice with weights\nBatch_size=512. Time in ms') #  time processing over buffer_size.
    plt.xlabel('Buffer size')
    plt.ylabel('Processing time')
    plt.grid('both')
    plt.show()

