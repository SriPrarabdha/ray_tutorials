import os
import io
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Generator
import json
from PIL import Image
from google.cloud import storage


class GCSDatasetReader:
    """
    A lazy dataset reader for Google Cloud Storage.
    Reads images from a GCS bucket without loading all into memory at once.
    """
    def __init__(self, gcs_path: str, pattern: str = None):
        """
        Args:
            gcs_path: GCS path like 'gs://bucket-name/folder/' or 'gs://bucket-name/folder/subfolder/'
            pattern: File extension filter (e.g., '.jpg', '.png'). None means all files.
        """
        self.gcs_path = gcs_path.rstrip('/')
        self.pattern = pattern
        self.bucket_name, self.prefix = self._parse_gcs_path(gcs_path)
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)
        self.blob_list = []
        self._load_blob_list()
    
    def _parse_gcs_path(self, gcs_path: str) -> tuple:
        """Parse GCS path into bucket and prefix"""
        if not gcs_path.startswith('gs://'):
            raise ValueError("GCS path must start with 'gs://'")
        
        path_parts = gcs_path[5:].split('/', 1)
        bucket_name = path_parts[0]
        prefix = path_parts[1] if len(path_parts) > 1 else ''
        
        return bucket_name, prefix
    
    def _load_blob_list(self):
        """Get list of blobs (files) from the GCS bucket"""
        print(f"Scanning GCS bucket: {self.bucket_name}/{self.prefix}")
        
        blobs = self.client.list_blobs(self.bucket_name, prefix=self.prefix)
        
        for blob in blobs:
            # Skip directories (blobs ending with /)
            if blob.name.endswith('/'):
                continue
            
            # Apply pattern filter if specified
            if self.pattern is None or blob.name.endswith(self.pattern):
                self.blob_list.append(blob.name)
        
        print(f"Found {len(self.blob_list)} files in GCS")
    
    def _download_image_from_gcs(self, blob_name: str) -> np.ndarray:
        """Download and decode image from GCS"""
        blob = self.bucket.blob(blob_name)
        image_bytes = blob.download_as_bytes()
        
        # Open image with PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array
    
    def iter_rows(self) -> Generator[Dict, None, None]:
        """Generator that yields rows (dict format) one at a time"""
        for idx, blob_name in enumerate(self.blob_list):
            try:
                image = self._download_image_from_gcs(blob_name)
                
                yield {
                    'id': idx,
                    'image': image,
                    'filename': os.path.basename(blob_name),
                    'gcs_path': f"gs://{self.bucket_name}/{blob_name}"
                }
            except Exception as e:
                print(f"Error loading {blob_name}: {e}")
                continue
    
    def map(self, func, num_workers: int = 4):
        """
        Apply a function to each row in parallel.
        Returns a ProcessedDataset object.
        """
        return ProcessedDataset(self, func, num_workers)


class ProcessedDataset:
    """
    Represents a lazily-processed dataset that can be written to GCS.
    """
    def __init__(self, source_dataset: GCSDatasetReader, map_func, num_workers: int = 4):
        self.source_dataset = source_dataset
        self.map_func = map_func
        self.num_workers = num_workers
        self.results = []
    
    def _process_all(self):
        """Process all rows using multiprocessing"""
        if self.results:  # Already processed
            return
        
        print(f"Loading images from GCS...")
        rows = list(self.source_dataset.iter_rows())
        
        print(f"Processing {len(rows)} images with {self.num_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(self.map_func, row): row for row in rows}
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    self.results.append(result)
                    completed += 1
                    if completed % 10 == 0:
                        print(f"  Processed {completed}/{len(rows)} images...")
                except Exception as e:
                    print(f"Error processing row: {e}")
        
        print(f"Completed processing {len(self.results)} images")
    
    def write_to_gcs(self, output_gcs_path: str, save_format: str = 'npz'):
        """
        Write processed results to GCS.
        
        Args:
            output_gcs_path: GCS path like 'gs://bucket-name/processed-images/'
            save_format: 'npz' (numpy), 'png', or 'jpg'
        """
        self._process_all()
        
        bucket_name, prefix = self._parse_gcs_path(output_gcs_path)
        prefix = prefix.rstrip('/') + '/'
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        print(f"Writing {len(self.results)} processed images to {output_gcs_path}")
        
        # Save each processed image
        for idx, result in enumerate(self.results):
            blob_name = f"{prefix}processed_{idx}.{save_format}"
            blob = bucket.blob(blob_name)
            
            if save_format == 'npz':
                # Save as compressed numpy
                buffer = io.BytesIO()
                np.savez_compressed(
                    buffer,
                    processed_image=result['processed_image'],
                    original_id=result['original_id']
                )
                buffer.seek(0)
                blob.upload_from_file(buffer, content_type='application/octet-stream')
            
            elif save_format in ['png', 'jpg']:
                # Save as image
                img = Image.fromarray(result['processed_image'].astype('uint8'))
                buffer = io.BytesIO()
                img.save(buffer, format=save_format.upper())
                buffer.seek(0)
                blob.upload_from_file(buffer, content_type=f'image/{save_format}')
            
            if (idx + 1) % 10 == 0:
                print(f"  Uploaded {idx + 1}/{len(self.results)} files...")
        
        # Save metadata
        metadata = {
            'num_images': len(self.results),
            'output_format': save_format,
            'source_bucket': self.source_dataset.bucket_name,
            'source_prefix': self.source_dataset.prefix
        }
        
        metadata_blob = bucket.blob(f"{prefix}metadata.json")
        metadata_blob.upload_from_string(
            json.dumps(metadata, indent=2),
            content_type='application/json'
        )
        
        print(f"✓ Successfully saved {len(self.results)} files to {output_gcs_path}")
    
    def _parse_gcs_path(self, gcs_path: str) -> tuple:
        """Parse GCS path into bucket and prefix"""
        if not gcs_path.startswith('gs://'):
            raise ValueError("GCS path must start with 'gs://'")
        
        path_parts = gcs_path[5:].split('/', 1)
        bucket_name = path_parts[0]
        prefix = path_parts[1] if len(path_parts) > 1 else ''
        
        return bucket_name, prefix


def process_image(row: dict) -> dict:
    """
    Process a single image from a dictionary format.
    Inverts the image and returns both processed and original data.
    """
    image = row['image']
    inverted_image = 255 - image
    return {
        'processed_image': inverted_image,
        'original_id': row.get('id', None),
        'original_filename': row.get('filename', None)
    }


def read_images(gcs_path: str, pattern: str = None) -> GCSDatasetReader:
    """
    Create a lazy reference to images in a GCS bucket.
    
    Args:
        gcs_path: GCS path like 'gs://your-bucket-name/raw-images/'
        pattern: File extension filter (e.g., '.jpg', '.png')
    
    Returns:
        GCSDatasetReader object
    """
    return GCSDatasetReader(gcs_path, pattern)


def main():
    """
    Main pipeline execution.
    
    Before running:
    1. Install: pip install google-cloud-storage pillow numpy
    2. Set up authentication:
       - export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
       OR
       - gcloud auth application-default login
    """
    
    # Configuration
    INPUT_GCS_PATH = "gs://your-bucket-name/raw-images/"
    OUTPUT_GCS_PATH = "gs://your-bucket-name/processed-images/"
    NUM_WORKERS = 4
    FILE_PATTERN = '.jpg'  # Filter for .jpg files, use None for all files
    OUTPUT_FORMAT = 'png'  # 'npz', 'png', or 'jpg'
    
    print("=" * 60)
    print("GCS DATA PIPELINE")
    print("=" * 60)
    
    # 1. Create a lazy reference to the dataset in GCS
    print("\n[1/3] Creating dataset reference from GCS...")
    ds = read_images(INPUT_GCS_PATH, pattern=FILE_PATTERN)
    
    # 2. Define the distributed transformation
    print("\n[2/3] Defining map transformation...")
    processed_ds = ds.map(process_image, num_workers=NUM_WORKERS)
    
    # 3. Trigger the computation by writing the results to GCS
    print("\n[3/3] Executing pipeline and writing results to GCS...")
    processed_ds.write_to_gcs(OUTPUT_GCS_PATH, save_format=OUTPUT_FORMAT)
    
    print("\n" + "=" * 60)
    print("✓ Pipeline complete!")
    print(f"  Input:  {INPUT_GCS_PATH}")
    print(f"  Output: {OUTPUT_GCS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()