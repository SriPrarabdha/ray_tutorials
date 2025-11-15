import ray
import numpy as np
import time

# connect to the running cluster
ray.init(address='auto')


# Doesn't read the whole data just creates a plan
ds = ray.data.read_images("gs://bucket-name/raw-images/")

print(ds)
#Output: Dataset(num_blocks = .. , num_rows = ... , schema = ={image: ArrowTensorType(...)})

@ray.remote
def process_image(row: dict) -> dict:
    image = row['image']
    time.sleep(1)
    inverted_image = 255 - image
    return {'processed_image' : inverted_image , "original_id": row.get('id', None)}

# This defines the computation graph. It is still LAZY . Nothing has happen yet.
processed_ds = ds.map(process_image)

# This call triggers the distributed computation:
# 1. Ray Data starts reading blocks of images from S3 in parallel.
# 2. It sends these blocks directly to `process_image` tasks on the cluster.
# 3. It gathers the results and writes them back to S3.

processed_ds.write_parquet("gs://bucket-name/processed-images/")


