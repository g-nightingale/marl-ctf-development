import multiprocessing as mp
import numpy as np
from mp_classes import ClassA, ClassB
from mp_runner import Runner

if __name__ == "__main__":
    num_instances = 8
    runner = Runner()
    results = runner.parallel(num_instances)
    print(results)