Here is a complete, clean, and professional `README.md` for your `cell2text` project. I've structured it to concisely explain the architecture, outline the pipeline, and provide the exact execution commands based on the scripts you provided.

***

```markdown
# 🧬 Cell2Text

> A Multimodal Large Language Model pipeline that translates single-cell transcriptomic data (gene expression) into comprehensive natural language descriptions.

Cell2Text bridges the gap between single-cell biology and natural language processing. By leveraging foundational models for both biology and language, it takes raw cell gene expression tokens, projects them into a language semantic space, and generates detailed biological descriptions of the cell (including cell type, tissue, disease state, and active pathways).

## 🏗️ Architecture

The model uses a three-stage encoder-projector-decoder architecture:
1. **Cell Encoder (Geneformer):** Processes rank-value encoded gene expression data into dense embeddings. (Frozen during training).
2. **Projector (MLP / Q-Former / Perceiver):** Bridges the modality gap, aligning the high-dimensional biological embeddings into the LLM's text embedding space.
3. **LLM Decoder (LLaMA-3.2):** An autoregressive language model that attends to the projected cell embeddings to generate natural language descriptions. (Fine-tuned using LoRA or Full-parameter via FSDP).

---

## 🚀 Getting Started

### Prerequisites
Ensure you have the required dependencies installed (e.g., `torch`, `transformers`, `scanpy`, `cellxgene_census`, `geneformer`, `peft`). 

*Note: You will need the pretrained Geneformer weights, a LLaMA tokenizer/model, and necessary ontology files (like `obo.json` and GMT pathway files).*

---

## 🛠️ Pipeline Details & Usage

### 1. Dataset Creation (`create_dataset.py`)
This script handles the end-to-end data processing pipeline directly from the **CELLxGENE Census**. 

**What it does:**
* Fetches primary human cell data, filtering out niche/proprietary assays.
* Applies a robust stratified sampling strategy to ensure balanced distributions across Tissues, Cell Types, Diseases, and Donors.
* Performs global normalization, Highly Variable Gene (HVG) selection, and AUCell pathway enrichment.
* Creates an intelligent 80/10/10 (Train/Val/Test) split that prevents donor leakage while guaranteeing category representation.
* Automatically generates accurate **Natural Language Descriptions** for each cell based on biological metadata and ontologies.
* Tokenizes the final dataset using Geneformer's `TranscriptomeTokenizer`.

**Execution:**
```bash
python data_preprocess/create_dataset.py \
    -N 100000 \
    -BS 1000 \
    -CS 1000
```
* `-N`: Target sample size (total cells to extract).
* `-BS`: Batch size for saving `.h5ad` files.
* `-CS`: Chunk size for processing AUCell pathway enrichment.

### 2. Training (`train_ddp.py` & `train_fsdp.py`)
The repository supports scalable distributed training using either **Distributed Data Parallel (DDP)** or **Fully Sharded Data Parallel (FSDP)**. 

**What they do:**
* **`train_ddp.py`:** Ideal for Parameter-Efficient Fine-Tuning (PEFT). It freezes the Geneformer encoder, applies LoRA to the LLaMA decoder, and fully trains the Projector layer. Includes custom checkpointing that separates LoRA adapters from projector weights to save space.
* **`train_fsdp.py`:** Designed for memory-intensive full-parameter fine-tuning. Wraps the entire model (or massive decoders) using PyTorch's FSDP with Mixed Precision and CPU Offloading. 
* *Both scripts support a `--mode sanity` flag to rapidly overfit on a small dataset to verify pipeline integrity.*

**Execution (DDP with LoRA):**
```bash
torchrun --nproc_per_node=<NUM_GPUS> cell2text_train/train_ddp.py \
    --mode full \
    --train_data_path /path/to/train \
    --val_data_path /path/to/val \
    --projector mlp \
    --batch_size_per_device 4 \
    --epochs 5
```

**Execution (FSDP):**
```bash
torchrun --nproc_per_node=<NUM_GPUS> cell2text_train/train_fsdp.py \
    --mode full \
    --train_data_path /path/to/train \
    --projector qformer
```

### 3. Evaluation (`run_evaluation.py` & `celltype_ontology_metric.ipynb`)
Evaluating multimodal biological text generation requires more than just standard NLP metrics.

**What it does:**
* Loads the trained projector and LoRA/FSDP checkpoints.
* Generates descriptions for the held-out test set in a distributed manner.
* **Standard NLP Metrics:** Computes BLEU (1-4) and ROUGE (1, 2, L) for exact n-gram matching.
* **Semantic Metrics:** Uses BERTScore (BioBERT & RoBERTa) to evaluate the biological semantic similarity of the generated text versus the ground truth.
* **Biological Ontology Metrics (`celltype_ontology_metric` logic):** Extracts the predicted cell types/diseases and computes an **Ontology Similarity Score**. Instead of strict string matching, this calculates the graph distance between the predicted cell type and the true cell type within the Cell Ontology (CL) or Disease Ontology (DOID) DAGs, providing a nuanced biological accuracy score.

**Execution:**
```bash
python cell2text_eval/run_evaluation.py \
    --checkpoint_path /path/to/checkpoint \
    --test_data_path /path/to/test_dataset \
    --tokenizer_path /path/to/llama_tokenizer \
    --geneformer_path /path/to/geneformer \
    --llama_path /path/to/llama \
    --projector mlp \
    --batch_size 8 \
    --use_bertscore True \
    --use_comprehensive_metrics True
```
*(Add `--load_from_fsdp` if you are evaluating a checkpoint generated by the FSDP script).*

---

## 📂 Repository Structure
* `/data_preprocess`: Scripts for fetching from CELLxGENE, mapping ontologies, and tokenizing data.
* `/cell2text_model`: Contains the core architecture (`model.py`), merging Geneformer, the Projector classes, and LLaMA.
* `/cell2text_train`: PyTorch distributed training loops (`train_ddp.py`, `train_fsdp.py`).
* `/cell2text_eval`: Inference scripts and complex biological/NLP evaluation metrics.
```
