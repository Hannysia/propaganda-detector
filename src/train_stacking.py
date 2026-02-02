import os
import pandas as pd
import json
import optuna
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from huggingface_hub import HfApi, hf_hub_download
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_MODELS = [
    "roberta-base",
    "microsoft/deberta-v3-base",
    "xlnet-base-cased",
    "unitary/toxic-bert"
]

HF_DATASET_REPO = "hannusia123123/propaganda-detector-dataset"
HF_STACKING_REPO = "hannusia123123/propaganda-stacking-catboost"
HF_TOKEN = os.getenv("HF_TOKEN")

class PropagandaStacker:
    def __init__(self):
        self.api = HfApi()
        self.le = LabelEncoder()
        
    def load_and_merge(self, prefix="oof"):
        print(f"\nDownloading & Merging {prefix.upper()} predictions...")
        merged_df = None
        
        logit_cols = [f"logit_{i}" for i in range(14)]
        
        base_merge_keys = ['article_id', 'start_offset', 'true_label']
        
        for model_name in BASE_MODELS:
            safe_name = model_name.replace("/", "-")
            filename = f"{prefix}_{safe_name}.csv"
            
            try:
                path = hf_hub_download(
                    repo_id=HF_DATASET_REPO, 
                    filename=filename, 
                    repo_type="dataset",
                    token=HF_TOKEN
                )
                df = pd.read_csv(path)
                
                merge_keys = base_merge_keys.copy()
                if 'fold' in df.columns:
                    merge_keys.append('fold')

                cols_to_use = merge_keys + logit_cols
                df_subset = df[cols_to_use].copy()
                
                rename_map = {col: f"{safe_name}_{col}" for col in logit_cols}
                df_subset = df_subset.rename(columns=rename_map)
                
                if merged_df is None:
                    merged_df = df_subset
                else:
                    common_keys = [k for k in merge_keys if k in merged_df.columns]
                    merged_df = merged_df.merge(df_subset, on=common_keys)
                    
                print(f"  ✅ Loaded {safe_name} ({len(df)} rows)")
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                raise e
        
        return merged_df

    def prepare_data(self, df):
        feature_cols = [c for c in df.columns if "logit" in c]
        
        X = df[feature_cols]
        try:
            y = self.le.transform(df['true_label'])
        except:
            y = self.le.fit_transform(df['true_label'])
            
        return X, y, feature_cols

    def find_best_params(self, X, y, n_trials=15, search_space=None):

        print(f"\nStarting Optuna Tuning ({n_trials} trials)...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2,
            random_state=42,
            stratify=y,
            shuffle=True
        )

        def objective(trial):
            if search_space is None:
                params = {
                    'iterations': trial.suggest_int('iterations', 500, 1500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'depth': trial.suggest_int('depth', 4, 8),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'random_strength': trial.suggest_float('random_strength', 1e-9, 10),
                }
            else:
                params = search_space(trial)

            fixed_params = {
                'loss_function': 'MultiClass',
                'eval_metric': 'TotalF1',
                'random_seed': 42,
                'verbose': False,
                'allow_writing_files': False,
                'auto_class_weights': 'Balanced'
            }
            params.update(fixed_params)
            
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
            
            preds = model.predict(X_val)
            return f1_score(y_val, preds, average='macro')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"🎉 Best Params: {study.best_params}")
        return study.best_params
    

    def train_final_model(self, X_train, y_train, X_val, y_val, params=None):
        print("\n🚀 Training Final Stacking Model...")
        
        final_params = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'loss_function': 'MultiClass',
            'eval_metric': 'TotalF1',
            'random_seed': 42,
            'auto_class_weights': 'Balanced',
            'verbose': 100
        }
        
        if params:
            final_params.update(params)
        
        model = CatBoostClassifier(**final_params)
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100
        )
        
        return model

    def evaluate_and_save(self, model, X_test, y_test):
        print("\nFINAL HOLDOUT RESULTS:")
        class_names = list(self.le.classes_)
        
        final_preds = model.predict(X_test)
        
        print(classification_report(y_test, final_preds, target_names=class_names, digits=4))
        f1 = f1_score(y_test, final_preds, average='macro')
        print(f"FINAL ENSEMBLE F1: {f1:.4f}")

        print("\nSaving artifacts...")
        os.makedirs("stacking_artifacts", exist_ok=True)
        model.save_model("stacking_artifacts/catboost.cbm")
        
        with open("stacking_artifacts/labels.json", "w") as f:
            json.dump(class_names, f)
            
        if HF_TOKEN:
            try:
                self.api.create_repo(repo_id=HF_STACKING_REPO, exist_ok=True)
                self.api.upload_folder(
                    folder_path="stacking_artifacts",
                    repo_id=HF_STACKING_REPO,
                    repo_type="model",
                    commit_message=f"CatBoost Ensemble F1={f1:.4f}"
                )
                print("✅ Uploaded to Hugging Face!")
            except Exception as e:
                print(f"⚠️ Upload failed: {e}")
