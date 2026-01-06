# 🌟 MWP: MLLM-Guided Weak Prior for Cross-Modal Retrieval

<p align="center">
  <img src="assets/banner.png" width="92%" alt="MWP Banner"/>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-datasets">Datasets</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-data-preparation">Data Preparation</a> •
  <a href="#-training">Training</a> •
  <a href="#-evaluation">Evaluation</a> •
  <a href="#-results">Results</a>
</p>

---

## ✨ Overview

**MWP** is a weak-prior guided cross-modal retrieval framework that leverages *noisy MLLM-generated image descriptions* to build a **weak structural prior (WPE)** and then trains a **deployable retriever (GMR)** under this guidance.

🔑 Key idea:
- 🌤️ **WPE (Weak Prior Estimator)** learns / produces a **soft prior similarity** \( S^{T} \) from *(description, caption)* pairs.  
- 🔎 **GMR (Granular Multi-Retriever)** learns **retrieval logits** \( S^{S} \) from *(caption, image)* pairs and aligns with WPE priors.

📌 Training is **two-stage**:
1. **Stage-1:** Train **WPE** (prior modeling)  
2. **Stage-2:** Train **GMR** guided by frozen WPE (prior alignment)

---

## 📚 Datasets

We evaluate MWP on four widely-used cross-modal retrieval datasets:

- **Wikipedia**
- **Pascal Sentence**
- **NUS-WIDE-10k**
- **XMediaNet**

> 🧩 Notes  
> - Each sample contains: image, caption/text, and a category label (for category-aware training/evaluation).  
> - We additionally use **MLLM-generated image descriptions** (noisy) to train or infer WPE priors.

---

## 🧰 Installation

### ✅ Requirements
- Python >= 3.9
- PyTorch >= 1.13 (recommended 2.x)
- transformers
- tqdm, numpy, pandas

### ⚙️ Setup

```bash
conda create -n mwp python=3.10 -y
conda activate mwp

pip install -r requirements.txt
