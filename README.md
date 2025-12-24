# 🕵️ Propaganda Detection using BERT

This project aims to detect and classify propaganda techniques in news articles using NLP (SemEval-2020 Task 11).

## 🚀 MLOps Pipeline
1.  **Data Processing:** Custom parser to handle article-level splits.
2.  **Training:** Fine-tuning `bert-base-uncased` using Hugging Face Transformers.
3.  **Tracking:** Experiment tracking with Weights & Biases.
4.  **Deployment:** Interactive Gradio app hosted on Hugging Face Spaces.

## 🛠️ Project Structure
* `src/` - Data processing and utility scripts.
* `deployment/` - Code for the Gradio interface.
* `notebooks/` - EDA and experiments.
* `train.py` - Main training script suitable for Kaggle/Local execution.

## 📦 How to Run
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Setup `.env` file with your API keys (`WANDB_API_KEY`, `HF_TOKEN`).
3.  Run training:
    ```bash
    python train.py
    ```