import numpy as np
from mp_classes import ClassA, ClassB
import multiprocessing as mp

class Runner:
    def __init__(self, foobar=1.0):
        self.foobar = 1.0

    def parallel(self, num_instances):

        cb = ClassB()
        
        # Create a Pool object with 8 processes
        with mp.Pool(processes=num_instances) as pool:
            async_results = []

            # Use apply_async to run the 'random_numbers' function 8 times in parallel
            for _ in range(num_instances):
                arr = np.random.rand(10)
                async_result = pool.apply_async(cb.process, (arr,))
                async_results.append(async_result)

            # Collect the results
            results = [async_result.get() for async_result in async_results]

        return results