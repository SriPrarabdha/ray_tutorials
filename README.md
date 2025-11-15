
# A practical workshop on scaling ML workflows with the Ray ecosystem


## 🚀 Overview

Modern AI systems are no longer bottlenecked by *models* — they are bottlenecked by **infrastructure**. Training models, processing terabytes of data, deploying LLMs, and orchestrating GPU clusters all require tooling that simplifies distributed systems.

**Ray** is that tooling.

This repository contains everything used in my **60-minute workshop on AI Infrastructure with Ray**.
Each folder contains:

* A **baseline implementation** using traditional Python / PyTorch / multiprocessing
* A **Ray-powered implementation** showing how the same workflow becomes scalable, cleaner, and fault-tolerant

If you are new to Ray, this repo will help you understand not just APIs —
but *how Ray changes the way ML engineers build systems*.


# 📚 What You’ll Learn

Inside this repo you will learn how to:

### 🔹 **Scale Python code to clusters without rewriting it**

Ray Tasks & Actors turn normal Python functions into distributed workloads effortlessly.

### 🔹 **Process huge datasets with Ray Data**

Load, stream, preprocess, and batch terabytes of text or images using a completely Pythonic interface.

### 🔹 **Train models across multiple GPUs & nodes**

Use Ray Train to scale PyTorch/HuggingFace/PEFT models with fault tolerance and automatic checkpointing.

### 🔹 **Serve ML models (including LLMs) in production**

Ray Serve lets you deploy, autoscale, route, batch, and version models — including vLLM deployments.

### 🔹 **Run high-throughput LLM inference**

Use vLLM with Ray Serve for blazing-fast, production-grade LLM inference.

---

# 🗂 Repository Structure

```
ray_tutorials/
├── ray_core/
│   ├── baseline              # Standard Python multiprocessing / threading examples
│   ├── ray_tasks             # Same examples rewritten using Ray Tasks & Actors
│   └── ray_actors            # How to run on real Ray clusters (VMs, LAN, K8s)
│
├── ray_data/
│   ├── baseline              # Pandas, plain Python data pipelines
│   └── ray_version           # Ray Data: distributed loading, batching, streaming
│
├── ray_train/
│   ├── baseline              # Single-GPU PyTorch training (DDP optional)
│   └── ray_version           # Ray Train distributed training, FT, checkpoints
│
├── ray_serve/
│   ├── baseline              # Simple Flask/FastAPI serving patterns
│   └── ray_version           # Ray Serve deployments, autoscaling, routing, batching
│
├── ray_tune/
│   └── examples              # will be added soon
│
└── vllm_examples/
    
```

Every module includes:

✔ **Baseline Python code**
✔ **Ray implementation**
✔ **Explanations + comments**
✔ **Cluster-ready examples**

---

# 🧩 Who Is This For?

### 🎓 **Students & Researchers**

* Learn how to scale experiments without rewriting everything
* Build reproducible ML pipelines
* Run multi-GPU training in your lab or on the cloud

### 🛠 **ML Engineers**

* Build data pipelines, training pipelines, and serving pipelines
* Turn your laptop code into distributed code
* Deploy LLMs with autoscaling and batching

### 🏢 **Tech Teams / Startups**

* Build production ML infra without managing complex distributed systems
* Replace 5–6 tools with a unified Ray-based workflow
* Save engineering time and avoid infrastructure glue code

---

# 🏗 Philosophy of This Workshop

This workshop is built around a simple idea:

> **“Distributed ML should not require learning distributed systems.”**

Ray lets you scale your code using:

* your **existing Python functions**
* your **existing PyTorch models**
* your **existing HuggingFace workflows**
* your **existing serving patterns**

No MPI.
No Kubernetes YAML.
No Spark jobs.
No complicated Docker setups (unless you want them).

You get the power of a large-scale distributed system
**with the simplicity of standard Python.**

# 🤝 Contributions

Feel free to open issues or PRs if you have improvements, bug fixes, or new examples.

---

# ⭐ Acknowledgments

This workshop and repo were created as part of a technical session at **UbuCon India 2025**, showing how open-source AI infrastructure enables even small teams to operate like large ML organizations.

---
