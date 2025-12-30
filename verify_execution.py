from tools.eeg_experiment import run_eeg_experiment
import json
from pprint import pprint

def test_run():
    print("🚀 Starting Smoke Test: run_eeg_experiment (dry_run=False)...")
    
    # 最小構成のオーバーライド設定
    overrides = {
        "subject_ids": [1],   # 被験者1のみ
        "epochs": 2,          # テスト用に2エポックのみ (動作確認用)
        "batch_size": 16,     # メモリ負荷を軽く
        "n_splits": 2,        # 交差検証の分割数も最小に
        # 必要であれば "cuda": True などを追加（デフォルト設定による）
    }

    try:
        # 実実行
        result = run_eeg_experiment(
            overrides=overrides,
            run_label="smoke_verification",
            dry_run=False,  # ここが重要：実際にサブプロセスを起動
            timeout=600     # 10分でタイムアウト（安全性のため）
        )

        print("\n✅ Execution Completed.")
        print("-" * 40)
        
        # 結果の検証ポイント
        print(f"📂 Run Directory: {result.get('run_dir')}")
        print(f"🔢 Exit Code:    {result.get('exit_code')}")
        
        # メトリクス抽出の確認（これが最も重要）
        metrics = result.get("metrics", {})
        print(f"📊 Metrics Found: {metrics}")
        
        if "overall_mean_acc" in metrics:
            print(f"✨ SUCCESS: overall_mean_acc extracted -> {metrics['overall_mean_acc']}")
        else:
            print("⚠️ WARNING: overall_mean_acc NOT found. Check logs below.")
            print("Tail of stdout:")
            print(result.get("stdout_tail"))

    except Exception as e:
        print(f"❌ Error during execution: {e}")

if __name__ == "__main__":
    test_run()