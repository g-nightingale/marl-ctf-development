import multiprocessing as mp
import torch
import numpy as np
import timeit

# Define the function to generate a numpy array of 100 random numbers and add the value of 'c'
def random_numbers(n, c):
    return np.random.rand(n) + c

def synchronous(num_instances, n_numbers, c_value):
    sync_results = []
    for i in range(num_instances):
        sync_results.append(random_numbers(n_numbers, c_value))

    return sync_results

def parallel(num_instances, n_numbers, c_value):

    # Create a Pool object with 8 processes
    with mp.Pool(processes=num_instances) as pool:
        async_results = []

        # Use apply_async to run the 'random_numbers' function 8 times in parallel
        for _ in range(num_instances):
            async_result = pool.apply_async(random_numbers, (n_numbers, c_value))
            async_results.append(async_result)

        # Collect the results
        results = [async_result.get() for async_result in async_results]

    return results

if __name__ == '__main__':
    n_runs = 5
    n_numbers = 10000000
    num_instances = 50
    c_value = 5  # The value of 'c' to be added to the generated numbers

    results1 = synchronous(num_instances, n_numbers, c_value)
    results2 = parallel(num_instances, n_numbers, c_value)

    wrapped_synchronous = lambda: synchronous(num_instances, n_numbers, c_value)
    wrapped_parallel  = lambda: parallel(num_instances, n_numbers, c_value)

    # Run the function 100 times (default is 1,000,000) and calculate the average execution time
    print('Running synchronous...')
    execution_time_synchronous = timeit.timeit(wrapped_synchronous, number=n_runs) / 100
    print('Running parallel...')
    execution_time_parallel = timeit.timeit(wrapped_parallel, number=n_runs) / 100

    print(f"Synchronous took: {execution_time_synchronous:.8f} seconds to run on average.")
    print(f"Parallel took: {execution_time_parallel:.8f} seconds to run on average.")

    #Synchronous took: 0.13367342 seconds to run on average.
    #Parallel took: 1.14454067 seconds to run on average.