import os
import wandb
import random
import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv

def setup_environment():
    """
        Configures the keys and environment (Kaggle or Local).
        Returns the data path and HF token.
    """
    load_dotenv()
    
    hf_token = None
    
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        print("Detected Kaggle Environment ☁️")
        data_path = "/kaggle/working/dataset.csv"
        
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            
            # Login Hugging Face
            hf_token = user_secrets.get_secret("HF_TOKEN")
            from huggingface_hub import login
            login(token=hf_token)
            
            # Login W&B
            wandb_key = user_secrets.get_secret("WANDB_API_KEY")
            wandb.login(key=wandb_key)
            
        except Exception as e:
            print(f"⚠️ Auth warning: {e}")
            
    else:
        print("Detected Local Environment 🏠")
        data_path = "data/processed/dataset.csv"
        hf_token = os.getenv("HF_TOKEN")
        wandb.login()
        
    return data_path, hf_token



def print_distribution(df, name):
    """Displays statistics on class distribution in the dataset"""
    print(f"\n📊 --- {name} Set Statistics ---")
    print(f"Articles: {df['article_id'].nunique()} unique articles")
    print(f"Sentences: {len(df)} total sentences")
    
    counts = df['label'].value_counts()
    percs = df['label'].value_counts(normalize=True).mul(100).round(2)
    
    dist_df = pd.concat([counts, percs], axis=1, keys=['Count', 'Percent %'])
    print(dist_df)



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")