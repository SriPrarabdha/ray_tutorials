import os
import time
import json
import numpy as np
import ray

ray.init(address="auto") 

NUM_CPUS_PER_TASK = 1     
NUM_GPUS_PER_TASK = 0.25  
NUM_CPUS_PER_ACTOR = 0.5  
CHECKPOINT_FILE = "pixel_counter_checkpoint.json"

@ray.remote(num_cpus=NUM_CPUS_PER_ACTOR, max_restarts=-1, max_task_retries=-1)
class PixelCounter:
    def __init__(self, checkpoint_path: str = CHECKPOINT_FILE):
        self.total_pixels = 0
        self.checkpoint_path = checkpoint_path
        self._load_checkpoint()

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r") as f:
                    state = json.load(f)
                    self.total_pixels = state.get("total_pixels", 0)
                print(f"[PixelCounter] Restored checkpoint: {self.total_pixels} pixels")
            except Exception as e:
                print(f"[PixelCounter] Failed to load checkpoint: {e}")

    def _save_checkpoint(self):
        try:
            with open(self.checkpoint_path, "w") as f:
                json.dump({"total_pixels": self.total_pixels}, f)
            # Optionally sync to shared storage (NFS/S3) here
        except Exception as e:
            print(f"[PixelCounter] Failed to save checkpoint: {e}")

    def add(self, num_pixels: int):
        self.total_pixels += num_pixels
        # Periodically checkpoint
        if self.total_pixels % 1000 == 0:
            self._save_checkpoint()

    def get_total(self) -> int:
        return self.total_pixels

    def checkpoint_now(self):
        """Force checkpoint to disk."""
        self._save_checkpoint()
        return self.total_pixels


@ray.remote(num_cpus=NUM_CPUS_PER_TASK,
            num_gpus=NUM_GPUS_PER_TASK,
            max_retries=3)
def process_images(image: np.ndarray, counter_actor: "ActorHandle") -> np.ndarray:
    """
    slow image kernel with random simulated failure.
    """
    import random
    if random.random() < 0.1:
        raise RuntimeError("Simulated task failure")

    # Increment total pixels processed
    counter_actor.add.remote(image.size)

    # Simulate a slow filter
    time.sleep(1)
    processed = 255 - image
    return processed


if __name__ == "__main__":
    num_images = 8
    images = [np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8) for _ in range(num_images)]

    counter = PixelCounter.remote()

    start_time = time.time()
    result_refs = [process_images.remote(img, counter) for img in images]

    # Retrieve results (ray.get waits for successful completion)
    try:
        results = ray.get(result_refs)
    except Exception as e:
        print(f"Some tasks failed even after retries: {e}")
        results = []

    end_time = time.time()

    # Save final checkpoint
    ray.get(counter.checkpoint_now.remote())

    total_pixels = ray.get(counter.get_total.remote())

    print("------------------------------------------------------------------")
    print(f"✅ Processed {len(results)} images in {end_time - start_time:.2f} seconds")
    print(f"✅ Total pixels processed (from actor): {total_pixels}")
    print("------------------------------------------------------------------")

    ray.shutdown()
