import ray
import numpy as np
import time

ray.init()

@ray.remote
class PixelCounter:
    def __init__ (self):
        self.total_pixels = 0

    def add (self, num_pixels: int):
        self.total_pixels += num_pixels

    def get_total(self) -> int:
        return self.total_pixels

@ray.remote
def process_images(image: np.ndarray , counter_actor: "ActorHandle") -> np.ndarray:
    '''A dummy slow image filter'''
    counter_actor.add.remote(image.size)
    time.sleep(1)
    return 255 - image

images = [np.random.randint(0, 255, (10, 10, 3)) for _ in range(8)]
image_size = images[0].size

counter = PixelCounter.remote()

start_time = time.time()

result_refs = [process_images.remote(img, counter) for img in images]
results = ray.get(result_refs)
end_time = time.time()

print(f"Processed {len(results)} images in {end_time - start_time:.2f} seconds")
print(f"total pixels in main script = {ray.get(counter.get_total.remote())}")
ray.shutdown()