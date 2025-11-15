"""
Production-Ready Ray Serve vLLM Deployment
Supports: Multi-GPU, Dynamic Scaling, Versioning, Rolling Updates, Load Balancing

Setup:
1. Start Ray cluster head node: ray start --head --port=6379
2. On worker nodes: ray connect <head_node_ip>:6379
3. Run this script on head node or remotely: python deploy.py
4. Access via: http://<head_node_ip>:8000/generate
"""

import ray
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95

class GenerateResponse(BaseModel):
    text: str
    version: str

# FastAPI app for HTTP interface
app = FastAPI(title="vLLM Inference API")

@serve.deployment(
    name="vllm-deployment",
    version="v1.0",  # Version for rolling updates
    num_replicas=2,  # Start with 2 replicas
    ray_actor_options={
        "num_gpus": 1,  # 1 GPU per replica
        "num_cpus": 4,   # CPUs per replica
    },
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,  # Scale up to 4 replicas
        "target_num_ongoing_requests_per_replica": 5,  # Trigger scaling
        "metrics_interval_s": 10,
        "look_back_period_s": 30,
        "smoothing_factor": 0.5,
        "downscale_delay_s": 300,  # Wait 5 min before scaling down
        "upscale_delay_s": 30,     # Quick scale up
    },
    max_concurrent_queries=10,  # Per replica
    health_check_period_s=10,
    health_check_timeout_s=30,
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(self):
        logger.info("Initializing vLLM model...")
        self.llm = LLM(
            model="Qwen/Qwen2.5-0.5B",
            tensor_parallel_size=1,  # 1 GPU per model instance
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )
        self.version = "v1.0"
        logger.info(f"Model loaded successfully. Version: {self.version}")

    @app.get("/health")
    def health(self):
        """Health check endpoint"""
        return {"status": "healthy", "version": self.version}

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(self, request: GenerateRequest):
        """Main inference endpoint"""
        try:
            logger.info(f"Received request: {request.prompt[:50]}...")
            
            # Configure sampling parameters
            params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            
            # Generate response
            output = self.llm.generate([request.prompt], params)
            generated_text = output[0].outputs[0].text
            
            logger.info(f"Generated {len(generated_text)} characters")
            
            return GenerateResponse(
                text=generated_text,
                version=self.version
            )
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise

    @app.get("/")
    def root(self):
        """Root endpoint with API info"""
        return {
            "service": "vLLM Inference API",
            "version": self.version,
            "model": "Qwen/Qwen2.5-0.5B",
            "endpoints": {
                "generate": "POST /generate",
                "health": "GET /health"
            }
        }


def deploy(head_node_ip: Optional[str] = None, update: bool = False):
    """
    Deploy or update the service
    
    Args:
        head_node_ip: IP address of Ray head node (e.g., "192.168.1.100")
                     If None, connects to local Ray cluster
        update: If True, performs rolling update of existing deployment
    """
    # Initialize Ray
    if head_node_ip:
        ray.init(address=f"ray://{head_node_ip}:10001")
        logger.info(f"Connected to Ray cluster at {head_node_ip}")
    else:
        ray.init(address="auto")  # Connect to existing cluster
        logger.info("Connected to local Ray cluster")
    
    # Start Ray Serve
    serve.start(
        detached=True,  # Service survives script exit
        http_options={
            "host": "0.0.0.0",  # Listen on all interfaces
            "port": 8000,
        }
    )
    
    if update:
        logger.info("Performing rolling update...")
        # For rolling update, just increment version
        # Ray Serve will gradually replace old replicas
    
    # Deploy the service
    deployment = VLLMDeployment.bind()
    serve.run(
        deployment,
        name="vllm-service",
        route_prefix="/",
    )
    
    logger.info("=" * 60)
    logger.info("Deployment successful!")
    logger.info(f"Service available at: http://<head_node_ip>:8000")
    logger.info("API Documentation: http://<head_node_ip>:8000/docs")
    logger.info("=" * 60)
    logger.info("\nExample usage:")
    logger.info('curl -X POST "http://<head_node_ip>:8000/generate" \\')
    logger.info('  -H "Content-Type: application/json" \\')
    logger.info('  -d \'{"prompt": "Once upon a time", "max_tokens": 50}\'')
    logger.info("=" * 60)


def rolling_update(head_node_ip: Optional[str] = None, new_version: str = "v1.1"):
    """
    Perform a rolling update with a new version
    
    Steps:
    1. Modify the deployment version and any parameters
    2. Call serve.run() again - Ray Serve handles the rolling update
    """
    if head_node_ip:
        ray.init(address=f"ray://{head_node_ip}:10001")
    else:
        ray.init(address="auto")
    
    serve.start(detached=True)
    
    # Update deployment with new version
    @serve.deployment(
        name="vllm-deployment",
        version=new_version,  # New version
        num_replicas=2,
        ray_actor_options={"num_gpus": 1, "num_cpus": 4},
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 4,
            "target_num_ongoing_requests_per_replica": 5,
        },
    )
    @serve.ingress(app)
    class VLLMDeploymentV2:
        def __init__(self):
            self.llm = LLM(model="Qwen/Qwen2.5-0.5B")
            self.version = new_version
        
        # ... same methods as before
    
    logger.info(f"Rolling update to {new_version}...")
    deployment = VLLMDeploymentV2.bind()
    serve.run(deployment, name="vllm-service", route_prefix="/")
    logger.info("Rolling update initiated!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy vLLM service on Ray cluster")
    parser.add_argument(
        "--head-ip",
        type=str,
        default=None,
        help="Ray head node IP address (e.g., 192.168.1.100)"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Perform rolling update of existing deployment"
    )
    parser.add_argument(
        "--new-version",
        type=str,
        default="v1.1",
        help="New version for rolling update"
    )
    
    args = parser.parse_args()
    
    if args.update:
        rolling_update(args.head_ip, args.new_version)
    else:
        deploy(args.head_ip, args.update)