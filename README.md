

# Context-Aware Hierarchical Taxonomy Generation for Scientific Papers via LLM-Guided Multi-Aspect Clustering

This repository contains the code and dataset used in our paper:
**"Context-Aware Hierarchical Taxonomy Generation for Scientific Papers via LLM-Guided Multi-Aspect Clustering"**.

## 📂 Dataset

The dataset TaxoBench is available in the directory:

```
./data/ground_new
```

## 🛠️ Installation

Create and activate a new conda environment:

```bash
conda create -n taxo_hier python=3.10
conda activate taxo_hier
```

Then install all required dependencies:

```bash
pip install -r requirements.txt
```

## 🔐 API Key Setup

Create a file named `Keys.py` in the root directory with the following content:

```python
# For LLMs and embedding generation
OPENAI_KEY = 'YOUR_OPENAI_KEY'  # Optional: use your OpenAI API key
EMBD_MODEL = "text-embedding-3-large"

# Alternatively, for local LLMs:
SELECT_MODEL = "llama"
MODEL_PATH = "PATH/TO/Llama-3.1-8B-Instruct"

# For Semantic Scholar search:
SEMANTIC_SCHOLAR_API_KEY = 'YOUR_SEMANTIC_SCHOLAR_KEY'

# Output directory for results:
SAVE_PATH = "../res"
```

## 🚀 Running the Pipeline

Once everything is set up, you can run the full taxonomy generation pipeline with:

```bash
python src/pipeline.py
```

This will perform multi-aspect clustering and generate hierarchical taxonomies for scientific papers.
