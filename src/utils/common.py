import os
import random
import pandas as pd
import numpy as np
from sklearn.utils import resample
import torch
from dotenv import load_dotenv



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