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
