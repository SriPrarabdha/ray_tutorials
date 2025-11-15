import ray
from ray import serve
from vllm import LLM, SamplingParams

ray.init()
serve.start()

@serve.deployment(ray_actor_options={"num_gpus": 1})
class VLLMDeployment:
    def __init__(self):
        self.llm = LLM(model="Qwen/Qwen2.5-0.5B")
        self.params = SamplingParams(max_tokens=100)

    def __call__(self, prompt: str):
        output = self.llm.generate([prompt], self.params)
        return output[0].outputs[0].text

VLLMDeployment.deploy()


