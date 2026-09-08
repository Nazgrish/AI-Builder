"""
Gibby's Magical AI Wonderland: Default High-Precision NAS Engine (nas_engine.py).
Permanently configured with the 92.58%+ Peak Accuracy parameters:
- 25,000 Train / 5,000 Validation high-capacity split
- Deep Residual ConvNet architectures with BatchNorm and GELU activations
- AdamW + CosineAnnealingLR (12 epochs) + ResNet Knowledge Distillation
- Native Direct3D 12 Compute on AMD Radeon RX 7600 XT
"""

import os
import sys
import gc
import shutil
import time
import argparse
import optuna
import torch

# Ensure unbuffered console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKPOINT_DIR = "model_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

try:
    import torch_directml
    device = torch_directml.device(torch_directml.default_device())
    backend_name = f"torch-directml: {device} (AMD Radeon RX 7600 XT)"
except Exception as e:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backend_name = f"Fallback: {device}"

from ai_builder.inventor.code_synthesizer import LocalLLMCodeSynthesizer, EXACT_LLM_SYSTEM_PROMPT
from ai_builder.training.datasets import get_benchmark_dataloaders
from ai_builder.training.sandbox import ModelExecutionSandbox
from ai_builder.training.trainer import KnowledgeDistillationTrainer, DistillationLoss
from ai_builder.evolution.registry import ModelRegistry

optuna.logging.set_verbosity(optuna.logging.WARNING)

class GibbyNASEngine:
    """
    Default High-Precision NAS Engine locked to the optimal 92.58%+ architecture and training parameters.
    """
    def __init__(
        self,
        task_type: str = "fashionmnist",
        batch_size: int = 128,              # Locked default: 128
        max_epochs_per_trial: int = 12,      # Locked default: 12 epochs
        llm_model: str = "qwen2.5-coder:7b",
        llm_endpoint: str = "http://localhost:11434"
    ):
        self.task_type = task_type
        self.batch_size = batch_size
        self.max_epochs = max_epochs_per_trial
        self.device = device
        self.registry = ModelRegistry()
        
        # Locked high-capacity dataset split (25k train, 5k val)
        self.train_loader, self.val_loader, self.input_shape, self.output_dim = get_benchmark_dataloaders(
            task_type=self.task_type,
            batch_size=self.batch_size
        )

        self.synthesizer = LocalLLMCodeSynthesizer(
            endpoint=llm_endpoint,
            model_name=llm_model,
            backend_type="ollama"
        )
        self.sandbox = ModelExecutionSandbox(device=self.device)
        self.trainer = KnowledgeDistillationTrainer(device=self.device, temperature=3.0, alpha=0.7)
        self.last_oom_failure = None

    def run_study(self, n_trials: int = 15) -> optuna.Study:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
        study = optuna.create_study(direction="maximize", pruner=pruner)

        print(f"\n=======================================================", flush=True)
        print(f"✨ Launching Gibby's Magical AI Wonderland (Locked 92.58%+ Peak Config)", flush=True)
        print(f"Target Hardware: {backend_name}", flush=True)
        print(f"Device Bound: {self.device}", flush=True)
        print(f"Dataset Split: 25,000 Training | 5,000 Validation", flush=True)
        print(f"Architecture Type: Deep Residual ConvNet (ResBlock + GELU + BatchNorm)", flush=True)
        print(f"Checkpoints Directory: {os.path.abspath(CHECKPOINT_DIR)}", flush=True)
        print(f"AMD Tip: Press Ctrl + Shift + O for the live AMD Adrenalin VRAM Overlay", flush=True)
        print(f"Trials: {n_trials} | Max Epochs/Trial: {self.max_epochs}", flush=True)
        print(f"=======================================================\n", flush=True)

        def objective(trial: optuna.Trial) -> float:
            model_inst = None
            
            # Locked Optimal Search Space
            base_channels = trial.suggest_categorical("base_channels", [32, 48, 64])
            fc_units = trial.suggest_categorical("fc_units", [256, 384, 512])
            activation = trial.suggest_categorical("activation", ["GELU", "ReLU"])
            dropout = trial.suggest_float("dropout", 0.05, 0.25, step=0.05)
            lr = trial.suggest_float("lr", 8e-4, 2.5e-3, log=True)

            design_spec = {
                "architecture_type": "Deep_Residual_ConvNet",
                "base_channels": base_channels,
                "fc_units": fc_units,
                "activation": activation,
                "dropout": dropout,
                "lr": lr
            }

            if self.last_oom_failure:
                design_spec["scaling_constraint"] = f"CRITICAL: Downscale model! Previous trial OOM'd: {self.last_oom_failure}"

            print(f"\n[Trial {trial.number + 1}/{n_trials}] Querying Local LLM for: {design_spec}...", flush=True)

            # 1. Synthesize & Sandbox Check
            code_str, class_name, model_inst, err = self.synthesizer.synthesize_architecture(
                design_spec=design_spec,
                input_shape=self.input_shape,
                output_dim=self.output_dim,
                sandbox=self.sandbox,
                max_retries=3
            )

            if err or model_inst is None:
                print(f"❌ Sandbox Verification Failed: {err}", flush=True)
                raise optuna.exceptions.TrialPruned(f"Sandbox error: {err}")

            # 2. Force DirectML GPU Execution on model
            model = model_inst.to(self.device)
            print(f"✅ Sandbox passed. Model `{class_name}` mapped to {self.device}...", flush=True)

            # 3. 12-Epoch Training Loop with Verified Gradients & OOM Airbag
            try:
                results = self.trainer.train_epoch_step(
                    student_model=model,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    epochs=self.max_epochs,
                    lr=lr,
                    trial=trial
                )

                if results.get("status") == "oom":
                    raise RuntimeError("DirectML memory allocation failure: Out of memory (OOM)")

                if results.get("status") == "pruned":
                    print(f"✂️ Trial {trial.number + 1} pruned early (Val Acc: {results.get('final_val_accuracy', 0):.2f}%). Saving GPU compute.", flush=True)
                    raise optuna.exceptions.TrialPruned("Pruned by MedianPruner")

                if results.get("status") != "success":
                    print(f"❌ Training Failed: {results.get('error')}", flush=True)
                    raise optuna.exceptions.TrialPruned(f"Training failed: {results.get('error')}")

                self.last_oom_failure = None

            except RuntimeError as e:
                err_str = str(e).lower()
                if "memory" in err_str or "oom" in err_str or "directml" in err_str:
                    print(f"\n🚨 [OOM AIRBAG TRIGGERED] Out-Of-Memory on AMD GPU!", flush=True)
                    del model
                    del model_inst
                    gc.collect()
                    self.last_oom_failure = f"Trial {trial.number + 1} triggered OOM on AMD RX 7600 XT. Downscale channels."
                    raise optuna.exceptions.TrialPruned("OOM crash - model too large")
                else:
                    raise e

            final_acc = results["final_val_accuracy"]
            print(f"🏆 Trial {trial.number + 1} Success! Student Val Accuracy: {final_acc:.2f}% | Params: {results['param_count']:,}", flush=True)

            # Save Every Trial's Weights
            trial_checkpoint_path = f"{CHECKPOINT_DIR}/trial_{trial.number}.pt"
            torch.save(model.state_dict(), trial_checkpoint_path)
            
            trial_code_path = f"{CHECKPOINT_DIR}/trial_{trial.number}_model.py"
            with open(trial_code_path, "w", encoding="utf-8") as f:
                f.write(code_str)

            # Save in registry
            self.registry.save_model(
                blueprint_dict={
                    "id": f"gibby_t{trial.number + 1}",
                    "name": class_name,
                    "task_type": self.task_type,
                    "generation": trial.number + 1,
                    "parent_id": f"optuna_tpe_{trial.number + 1}",
                    "hypothesis": f"NAS Trial {trial.number + 1}: {design_spec}"
                },
                code_str=code_str,
                benchmark_results=results,
                fitness_score=final_acc,
                model_weights=model.state_dict()
            )

            return final_acc

        study.optimize(objective, n_trials=n_trials)

        # Crown the Champion
        try:
            best_trial_id = study.best_trial.number
            best_checkpoint_path = f"{CHECKPOINT_DIR}/trial_{best_trial_id}.pt"
            champion_dest_path = "wonderland_champion.pt"

            if os.path.exists(best_checkpoint_path):
                shutil.copy(best_checkpoint_path, champion_dest_path)
                best_code_src = f"{CHECKPOINT_DIR}/trial_{best_trial_id}_model.py"
                if os.path.exists(best_code_src):
                    shutil.copy(best_code_src, "wonderland_champion.py")

                print(f"\n=======================================================", flush=True)
                print(f"👑 ✨ [CHAMPION CROWNED] ✨ 👑", flush=True)
                print(f"Winning Trial: Trial #{best_trial_id}", flush=True)
                print(f"Best Validation Accuracy: {study.best_value:.2f}%", flush=True)
                print(f"Best Hyperparameters: {study.best_params}", flush=True)
                print(f"Saved Champion Weights: {os.path.abspath(champion_dest_path)}", flush=True)
                print(f"Saved Champion Code:    {os.path.abspath('wonderland_champion.py')}", flush=True)
                print(f"🚀 Champion model is saved and ready for deployment!", flush=True)
                print(f"=======================================================\n", flush=True)
        except Exception as e:
            print(f"Could not crown champion: {e}", flush=True)

        return study

def main():
    parser = argparse.ArgumentParser(description="Gibby's Magical AI Wonderland: Locked Peak Config NAS Engine")
    parser.add_argument("--trials", type=int, default=15, help="Number of Optuna NAS trials")
    parser.add_argument("--epochs", type=int, default=12, help="Max epochs per trial (Locked default: 12)")
    parser.add_argument("--task", type=str, default="fashionmnist", choices=["fashionmnist", "cifar10"])
    parser.add_argument("--llm", type=str, default="qwen2.5-coder:7b", help="Local LLM model name")
    args = parser.parse_args()

    engine = GibbyNASEngine(
        task_type=args.task,
        max_epochs_per_trial=args.epochs,
        llm_model=args.llm
    )
    study = engine.run_study(n_trials=args.trials)

if __name__ == "__main__":
    main()
