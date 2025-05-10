import multiprocessing
import time

from discrete_numeric import compute_circuits_for_step, k_f

# Test parameters (adjust as needed)
num_qubits = 200
steps_list = list(range(0, 51, 5))  # Small test workload
num_circuits = 100
noise_param = 0.3
ks = k_f(num_qubits)

worker_options = [max(1, multiprocessing.cpu_count()//2), multiprocessing.cpu_count(), multiprocessing.cpu_count()*2]
chunksize_options = [1, 2, 4, 8]

def benchmark_parallel_configurations(ks, steps_list, num_circuits, noise_param, worker_options, chunksize_options):
    results = []
    jobs = []
    for steps in steps_list:
        jobs.append((ks, steps, num_circuits, noise_param))
    for num_workers in worker_options:
        for chunksize in chunksize_options:
            print(f"\nTesting: num_workers={num_workers}, chunksize={chunksize}")
            start = time.time()
            with multiprocessing.Pool(processes=num_workers) as pool:
                _ = pool.map(compute_circuits_for_step, jobs, chunksize=chunksize)
            elapsed = time.time() - start
            print(f"Elapsed: {elapsed:.2f} seconds")
            results.append({'num_workers': num_workers, 'chunksize': chunksize, 'elapsed': elapsed})
    # Print summary
    print("\nSummary:")
    for r in results:
        print(f"Workers: {r['num_workers']}, Chunksize: {r['chunksize']}, Time: {r['elapsed']:.2f}s")
    return results

if __name__ == "__main__":
    benchmark_parallel_configurations(ks, steps_list, num_circuits, noise_param, worker_options, chunksize_options) 