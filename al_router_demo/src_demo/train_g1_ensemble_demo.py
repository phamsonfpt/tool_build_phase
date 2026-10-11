import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForObjectDetection

sys.path.append(str(Path(__file__).parent.parent))
from src.config import *
from src.data.bdd_dataset import BDD100KMiniDataset

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    print("=== START PHASE 1: DEEP ENSEMBLES (5 EXPERTS) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_model_path = G0_EXPERIMENT_DIR / "base_model"
    if not base_model_path.exists():
        print("Error: Base Model not found. Run Phase 0 first!")
        return

    # Configuration for 5 Experts with distinct seeds
    experts = {
        "model_A": {"seed": 42},
        "model_B": {"seed": 100},
        "model_C": {"seed": 2024},
        "model_D": {"seed": 777},
        "model_E": {"seed": 9999}
    }

    print("1. Loading Dataset (Dummy BDD100K)...")
    train_dataset = BDD100KMiniDataset(
        images_dir=IMAGES_DIR / "train",
        labels_file=LABELS_DIR / "train.json"
    )

    for model_name, config in experts.items():
        seed = config["seed"]
        print(f"\n--- Training Expert: {model_name} (Random Seed: {seed}) ---")
        
        # 1. Set Seed to control data shuffling differently for each expert
        set_seed(seed)
        
        # 2. Initialize DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        # 3. Load Base Model as the starting point
        model = AutoModelForObjectDetection.from_pretrained(base_model_path)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        
        # 4. Dummy Training Loop (1 Epoch per expert to prove pipeline)
        model.train()
        for epoch in range(1):
            print(f"   [Epoch 1/1] Processing batches for {model_name}...")
            for step, (images, targets) in enumerate(train_loader):
                # Fake loss for testing pipeline without complex Box matching
                dummy_loss = torch.tensor(1.5, requires_grad=True).to(device)
                optimizer.zero_grad()
                dummy_loss.backward()
                optimizer.step()
        
        # 5. Save Expert Model
        expert_dir = BASE_DIR / "experiments" / "G1_ensemble" / model_name
        os.makedirs(expert_dir, exist_ok=True)
        model.save_pretrained(expert_dir)
        print(f"   -> Expert saved at: {expert_dir}")

    print("\n=== END PHASE 1 ===")

if __name__ == "__main__":
    main()
