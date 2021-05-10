
"""
Parallel torch.flip operator benchmark, it depends on the
`benchmark_grain_size` branch from the PyTorch fork found at
https://github.com/andfoy/pytorch
"""

# Standard library imports
import json
import time
import os.path as osp

# Third-party imports
import cv2
import numpy as np
import torch
import tqdm

# Disable multithreading
# cv2.setNumThreads(1)
# torch.set_num_threads(1)


def timing():
    return {
        'float': {
            'cv2': [],
            'torch.flip': [],
            'flip (3000)': [],
            'flip (32768)': [],
            'indexing': []
        },
        'uint8': {
            'cv2': [],
            'torch.flip': [],
            'flip (3000)': [],
            'flip (32768)': [],
            'indexing': []
        }
    }


def size_entry():
    return {
        'horizontal': timing(),
        'vertical': timing()
    }


sizes = [
    (28, 28, 3),
    (128, 128, 3),
    (800, 640, 3),
    (1024, 768, 3),
    (1920, 1080, 3),
    (3840, 2160, 3)
]

sizes = [
    (7, 112, 3),
    (28, 28, 3),
    (112, 7, 3),

    (8, 2048, 3),
    (128, 128, 3),
    (2048, 8, 3),

    (5, 102400, 3),
    (800, 640, 3),
    (128000, 4, 3),

    (4, 196608, 3),
    (1024, 768, 3),
    (262144, 3, 3),

    (16, 129600, 3),
    (1920, 1080, 3),
    (230400, 9, 3),

    (16, 518400, 3),
    (3840, 2160, 3),
    (921600, 9, 3),
]

size_timing = []

func_transforms = {
    'cv2': lambda x: x,
    'torch.flip': lambda x: torch.from_numpy(x).permute(2, 0, 1).contiguous(),
    'flip (3000)': lambda x: torch.from_numpy(x).permute(2, 0, 1).contiguous(),
    'flip (32768)': lambda x: torch.from_numpy(x).permute(2, 0, 1).contiguous(),
    'indexing': lambda x: torch.from_numpy(x).permute(2, 0, 1).contiguous()
}

benchmark_funcs = {
    'vertical': {
        'cv2': lambda x, _: cv2.flip(x, 0),
        'torch.flip': lambda x, _: torch.flip_old(x, [1]),
        'flip (3000)': lambda x, _: torch.flip(x, [1], 3000),
        'flip (32768)': lambda x, _: torch.flip(x, [1], 32768),
        'indexing': lambda x, indices: x[:, indices, :]
    },
    'horizontal': {
        'cv2': lambda x, _: cv2.flip(x, 1),
        'torch.flip': lambda x, _: torch.flip_old(x, [2]),
        'flip (3000)': lambda x, _: torch.flip(x, [2], 3000),
        'flip (32768)': lambda x, _: torch.flip(x, [2], 32768),
        'indexing': lambda x, indices: x[:, :, indices]
    }
}

dtype_transforms = {
    'float': lambda x: x,
    'uint8': lambda x: (x * 255).astype(np.uint8)
}

index_generators = {
    'horizontal': lambda _, w: list(range(w))[::-1],
    'vertical': lambda h, _:  list(range(h))[::-1]
}

index_transforms = {
    'cv2': lambda x: x,
    'torch.flip': lambda x: x,
    'flip (3000)': lambda x: x,
    'flip (32768)': lambda x: x,
    'indexing': lambda x: torch.arange(len(x) - 1, -1, -1)
}

for size in sizes:
    H, W, _ = size
    size_sample = size_entry()

    for sample in tqdm.tqdm(range(100)):
        rand_input = np.random.rand(*size)
        for dtype in {'float', 'uint8'}:
            sample_input = dtype_transforms[dtype](rand_input)
            for func in {'cv2', 'torch.flip', 'flip (3000)', 'flip (32768)',
                         'indexing'}:
                bench_input = func_transforms[func](sample_input)
                for direction in {'vertical', 'horizontal'}:
                    indices = index_generators[direction](H, W)
                    bench_list = size_sample[direction][dtype][func]
                    bench_fn = benchmark_funcs[direction][func]

                    start_time = time.time()
                    indices = index_transforms[func](indices)
                    bench_fn(bench_input, indices)
                    end_time = time.time() - start_time
                    bench_list.append(end_time)

    size_timing.append({
        'size': size,
        'timing': size_sample
    })


with open('benchmark_results.json', 'w') as f:
    json.dump(size_timing, f)
