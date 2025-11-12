import ray
import numpy as np
import time

ray.init()

total_pixels_processed = 0

@ray.remote
def process_images(image: np.ndarray) -> np.ndarray:
    '''A dummy slow image filter'''
    time.sleep(1)

    # global total_pixels_processed
    # total_pixels_processed += image.size
    # print(f"Task prcessed {image.size} pixels")
    return 255 - image

images = [np.random.randint(0, 255, (10, 10, 3)) for _ in range(8)]

start_time = time.time()

result_refs = [process_images.remote(img) for img in images]
results = ray.get(result_refs)
end_time = time.time()

print(f"Processed {len(results)} images in {end_time - start_time:.2f} seconds")
print(f"total pixels in main script = {total_pixels_processed}")
ray.shutdown()