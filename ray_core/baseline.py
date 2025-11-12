import numpy as np
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool
import threading

def process_images(image: np.ndarray) -> np.ndarray:
    '''A dummy slow image filter'''
    time.sleep(1)
    return 255 - image

def sequential_processing(images):
    """Method 1: Sequential (baseline)"""
    print("Using Sequential Processing...")
    start_time = time.time()
    results = [process_images(img) for img in images]
    end_time = time.time()
    return results, end_time - start_time

def threadpool_processing(images, workers=4):
    """Method 2: ThreadPoolExecutor"""
    print(f"Using ThreadPoolExecutor with {workers} workers...")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process_images, images))
    end_time = time.time()
    return results, end_time - start_time

def processpool_processing(images, workers=4):
    """Method 3: ProcessPoolExecutor"""
    print(f"Using ProcessPoolExecutor with {workers} workers...")
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process_images, images))
    end_time = time.time()
    return results, end_time - start_time

def multiprocessing_pool(images, workers=4):
    """Method 4: multiprocessing.Pool"""
    print(f"Using multiprocessing.Pool with {workers} workers...")
    start_time = time.time()
    with Pool(processes=workers) as pool:
        results = pool.map(process_images, images)
    end_time = time.time()
    return results, end_time - start_time

def manual_threading(images, workers=4):
    """Method 5: Manual Threading"""
    print(f"Using Manual Threading with {workers} workers...")
    start_time = time.time()
    results = [None] * len(images)
    threads = []
    
    def worker(idx, img):
        results[idx] = process_images(img)
    
    # Process in batches equal to worker count
    for i in range(0, len(images), workers):
        batch_threads = []
        for j in range(i, min(i + workers, len(images))):
            t = threading.Thread(target=worker, args=(j, images[j]))
            t.start()
            batch_threads.append(t)
        
        for t in batch_threads:
            t.join()
    
    end_time = time.time()
    return results, end_time - start_time

def run_all_methods(images, workers=4):
    """Run all methods and compare"""
    print("=" * 60)
    print("COMPARING ALL PARALLELIZATION METHODS")
    print("=" * 60)
    
    methods = {
        'sequential': lambda: sequential_processing(images),
        'threadpool': lambda: threadpool_processing(images, workers),
        'processpool': lambda: processpool_processing(images, workers),
        'multiprocessing': lambda: multiprocessing_pool(images, workers),
        'threading': lambda: manual_threading(images, workers)
    }
    
    timings = {}
    for name, method in methods.items():
        results, elapsed = method()
        timings[name] = elapsed
        print(f"✓ {name:15s}: {elapsed:.2f} seconds ({len(results)} images)")
        print()
    
    print("=" * 60)
    print("SUMMARY:")
    fastest = min(timings.items(), key=lambda x: x[1])
    for name, elapsed in sorted(timings.items(), key=lambda x: x[1]):
        speedup = timings['sequential'] / elapsed
        marker = " ⭐ FASTEST" if name == fastest[0] else ""
        print(f"{name:15s}: {elapsed:6.2f}s  (speedup: {speedup:.2f}x){marker}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description='Parallel Image Processing with Multiple Methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python script.py --method sequential
  python script.py --method threadpool --workers 8
  python script.py --method all --images 16
  python script.py -m processpool -w 4 -n 12
        '''
    )
    
    parser.add_argument(
        '-m', '--method',
        type=str,
        choices=['sequential', 'threadpool', 'processpool', 'multiprocessing', 'threading', 'all'],
        default='threadpool',
        help='Parallelization method to use (default: threadpool)'
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='Number of workers/threads/processes (default: 4)'
    )
    
    parser.add_argument(
        '-n', '--images',
        type=int,
        default=8,
        help='Number of images to process (default: 8)'
    )
    
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Generate random images
    print(f"\nGenerating {args.images} random images (10x10x3)...")
    images = [np.random.randint(0, 255, (10, 10, 3)) for _ in range(args.images)]
    print(f"Workers/Threads: {args.workers}\n")
    
    # Method dispatcher
    methods = {
        'sequential': lambda: sequential_processing(images),
        'threadpool': lambda: threadpool_processing(images, args.workers),
        'processpool': lambda: processpool_processing(images, args.workers),
        'multiprocessing': lambda: multiprocessing_pool(images, args.workers),
        'threading': lambda: manual_threading(images, args.workers),
        'all': lambda: run_all_methods(images, args.workers)
    }
    
    if args.method == 'all':
        run_all_methods(images, args.workers)
    else:
        results, elapsed = methods[args.method]()
        print(f"\n{'='*60}")
        print(f"Processed {len(results)} images in {elapsed:.2f} seconds.")
        print(f"Method: {args.method}")
        print(f"Workers: {args.workers}")
        print(f"{'='*60}\n")

if __name__ == '__main__':
    main()