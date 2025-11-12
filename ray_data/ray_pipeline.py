import ray
import numpy as np
import time

# connect to the running cluster
ray.init(address='auto')

@ray.remote
def process_image(row: dict) -> dict:
    image = row['image']
    time.sleep(1)
    inverted_image = 255 - image
    return {'processed_image' : inverted_image , "original_id": row.get('id', None)}

ds = ray.data.read_images("gs://your-bucket-name/raw-images/")

processed_ds = ds.map(process_image)

processed_ds.write_parquet("gs://your-bucket-name/processed-images/")
