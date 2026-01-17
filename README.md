

# Context-Aware Hierarchical Taxonomy Generation for Scientific Papers via LLM-Guided Multi-Aspect Clustering

This repository contains the code and dataset used in our EMNLP 2025 paper:  
**Context-Aware Hierarchical Taxonomy Generation for Scientific Papers via LLM-Guided Multi-Aspect Clustering**  
([paper link](https://aclanthology.org/2025.emnlp-main.788/)).

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


---

## 📊 Evaluation of Generated Taxonomies

This repository includes tools to **evaluate automatically–generated taxonomy trees** against the **CS-TaxoBench ground-truth taxonomies** using structural and semantic metrics. 

### 📥 Evaluation Input Format

* **Ground truth directory**: a folder containing the expert-annotated CS-TaxoBench taxonomy trees.
* **Prediction directory**: a folder with model-generated taxonomy trees.
* Filenames **must match** between ground truth and prediction (e.g., `topic123.json` in both).
* Taxonomy files should be in Markdown format.

### 🚀 Running the Evaluation

Run the evaluation script with the following command:

```bash
python eval-3/eval_auto.py \
  --gt_dir ./ground_outline \
  --pred_dir ./res/test/heading \
  --output_path ./Evaluation_Scores.txt
```

#### Arguments

| Argument        | Description                                  |
| --------------- | -------------------------------------------- |
| `--gt_dir`      | Path to CS-TaxoBench ground-truth taxonomies |
| `--pred_dir`    | Path to model-generated taxonomy outputs     |
| `--output_path` | File to store evaluation scores and summary  |

### 📏 Evaluation Metrics

The evaluation scripts compute a variety of metrics including (but not limited to):

* **Structural similarity metrics** such as tree alignment scores and average degree similarity.
* **Semantic coherence metrics** based on level-order traversal comparisons and embedding-based soft recall.
* **Entity overlap / recall**, measuring lexical overlap between node labels.

These metrics aim to capture both **hierarchical organization** and **semantic fidelity** between generated and ground-truth taxonomies — addressing the core benchmarking goals of CS-TaxoBench. ([arXiv][1])

### 📄 Output Example

A typical evaluation output looks like:

```
ced: 18.0306 ± 9.5470
ced_score: 18.6409 ± 8.5281

Average Prediction Items: 9.3529 ± 5.6643
Average GroundTruth Items: 25.2288 ± 15.3408
Average Items Micro: 0.4565 ± 0.3125
Average Items Macro: 9.3529 ± 0.0000

Average Entity Recall: 0.0000 ± 0.0000
Average Heading Soft Recall P: 0.4499 ± 0.1333
Average Heading Soft Recall P union G: 0.9731 ± 0.0694
Average Heading Soft Recall: 0.4767 ± 0.1229

NMI: 0.4635 ± 0.1395
ARI: 0.1573 ± 0.1120
Purity: 0.4029 ± 0.1307
```

All metrics are reported as **mean ± standard deviation** across test instances.

---

### 📐 Metric Categories & Interpretation

#### 1. Structural Edit Distance Metrics

These metrics evaluate **hierarchical structure similarity** between generated and ground-truth taxonomies.

* **`ced`**
  Raw Structural Edit Distance* between two hierarchies
  → *Lower is better*

* **`ced_score`**
  Normalized for comparison across instances
  → *Higher is better*

These metrics focus on **tree topology**, independent of semantic similarity.

---

#### 2. Taxonomy Size & Coverage Statistics

These statistics reflect **content completeness and granularity**:

* **Average Prediction Items**
  Average number of nodes generated per taxonomy

* **Average GroundTruth Items**
  Average number of nodes in gold taxonomies

* **Average Items Micro**
  Ratio of predicted items to ground-truth items (micro-averaged)

* **Average Items Macro**
  Macro-level prediction size statistics

They help diagnose **under-generation** or **over-pruning** behaviors.

---

#### 3. Semantic Heading Alignment

**Soft Heading Recall** follows the definition in Appendix C.1 of the NAACL 2024 paper  
[*Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models*](https://aclanthology.org/2024.naacl-long.347.pdf#page=14).  

It computes semantic recall between generated and ground-truth headings using embedding-based *soft cardinality*, rather than exact string matching.

---

#### 4. Label Agreement Metrics

These metrics measure **alignment between predicted and gold taxonomy partitions**:

* **NMI (Normalized Mutual Information)**
* **ARI (Adjusted Rand Index)**
* **Purity**

They capture **global structural agreement** beyond local edit distances.
