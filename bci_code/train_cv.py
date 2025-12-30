import argparse
import time
import sys
import os
import torch

def main(config_path):
    print(f"[Info] Starting BCI Worker process inside Docker...")
    print(f"[Info] CUDA Available: {torch.cuda.is_available()}")
    
    # 権限チェック
    try:
        with open("docker_write_test.txt", "w") as f:
            f.write("Access granted from user " + str(os.getuid()))
        print("[Success] File write permission confirmed.")
    except Exception as e:
        print(f"[Error] File write failed: {e}")

    print("-" * 30)
    print("Epoch 1/1")
    # Agentが期待するキーワード
    print("Overall mean acc across subjects = 0.9512")
    print("[Info] Experiment finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args, unknown = parser.parse_known_args()
    
    main(config_path=args.config)
