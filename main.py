"""
Gibby's Magical AI Wonderland: Principled Neural Architecture Search (NAS) Engine.
Tailored for AMD Radeon RX 7600 XT with Local LLM Code Generation, Optuna TPE & ASHA Pruning,
Knowledge Distillation, and DirectML OOM Airbag Crash Protection.
"""

import os
import sys
import gc
import shutil
import time
import argparse
import optuna
import torch
from typing import Optional, Callable, Dict, Any, List

# Ensure safe encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKPOINT_DIR = "model_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

from ai_builder.backend.hardware import get_hardware_engine
from ai_builder.inventor.code_synthesizer import LocalLLMCodeSynthesizer
from ai_builder.training.datasets import get_benchmark_dataloaders
from ai_builder.training.sandbox import ModelExecutionSandbox
from ai_builder.training.trainer import KnowledgeDistillationTrainer
from ai_builder.evolution.registry import ModelRegistry

# Suppress verbose Optuna default logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

class GibbyNASOrchestrator:
    """
    Main Neural Architecture Search Engine for Gibby's Magical AI Wonderland.
    Includes OOM Crash Airbag Protection for AMD RX 7600 XT with DirectML.
    """
    def __init__(
        self,
        task_type: str = "fashionmnist",
        batch_size: int = 64,
        max_epochs_per_trial: int = 5,
        llm_model: str = "qwen2.5-coder:7b",
        llm_endpoint: str = "http://localhost:11434"
    ):
        self.task_type = task_type
        self.batch_size = batch_size
        self.max_epochs = max_epochs_per_trial
        
        self.hw_engine = get_hardware_engine()
        self.device = self.hw_engine.get_device()
        self.registry = ModelRegistry()
        
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
        self.trainer = KnowledgeDistillationTrainer(device=self.device, temperature=4.0, alpha=0.5)
        self.last_oom_failure: Optional[str] = None

    def run_nas_study(
        self,
        n_trials: int = 10,
        trial_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> optuna.Study:
        """
        Executes Optuna study with OOM Airbag protection and ASHA/MedianPruner.
        """
        pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
        study = optuna.create_study(direction="maximize", pruner=pruner)

        print(f"\n=======================================================")
        print(f"✨ Launching Gibby's Magical AI Wonderland NAS")
        print(f"Target Hardware: {self.hw_engine.backend_name} ({self.hw_engine.gpu_info.get('name')})")
        print(f"AMD Tip: Press Ctrl + Shift + O for the live AMD Adrenalin VRAM Overlay")
        print(f"Benchmark: {self.task_type.upper()} {self.input_shape} | Trials: {n_trials}")
        print(f"=======================================================\n")

        def objective(trial: optuna.Trial) -> float:
            model_inst = None
            
            # Hyperparameters
            conv_layers = trial.suggest_int("conv_layers", 2, 4)
            channels = trial.suggest_categorical("base_channels", [16, 32, 64])
            fc_units = trial.suggest_categorical("fc_units", [64, 128, 256])
            activation = trial.suggest_categorical("activation", ["GELU", "ReLU"])
            dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
            lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)

            design_spec = {
                "conv_layers": conv_layers,
                "base_channels": channels,
                "fc_units": fc_units,
                "activation": activation,
                "dropout": dropout,
                "lr": lr
            }

            if self.last_oom_failure:
                design_spec["scaling_constraint"] = f"CRITICAL: Downscale model! Previous trial OOM'd: {self.last_oom_failure}"

            print(f"\n[Trial {trial.number + 1}/{n_trials}] Querying Local LLM for: {design_spec}...")

            # 1. Synthesize & Sandbox Check
            code_str, class_name, model_inst, err = self.synthesizer.synthesize_architecture(
                design_spec=design_spec,
                input_shape=self.input_shape,
                output_dim=self.output_dim,
                sandbox=self.sandbox,
                max_retries=3
            )

            if err or model_inst is None:
                print(f"❌ Sandbox Verification Failed: {err}")
                raise optuna.exceptions.TrialPruned(f"Compilation error: {err}")

            model = model_inst.to(self.device)
            print(f"✅ Sandbox passed. Training Student `{class_name}` on {self.hw_engine.backend_name}...")

            # 2. Training Loop with OOM Airbag
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
                    raise RuntimeError("DirectML allocation failure: Out of memory (OOM)")

                if results.get("status") == "pruned":
                    print(f"✂️ Trial {trial.number + 1} pruned early. Saving GPU compute.")
                    raise optuna.exceptions.TrialPruned("Pruned by MedianPruner due to low accuracy")

                if results.get("status") != "success":
                    print(f"❌ Training Failed: {results.get('error')}")
                    raise optuna.exceptions.TrialPruned(f"Training failed: {results.get('error')}")

                self.last_oom_failure = None

            except RuntimeError as e:
                err_str = str(e).lower()
                if "memory" in err_str or "oom" in err_str or "directml" in err_str:
                    print(f"\n🚨 [OOM AIRBAG TRIGGERED] Out-Of-Memory on AMD GPU!")
                    del model
                    del model_inst
                    gc.collect()
                    self.last_oom_failure = f"Trial {trial.number + 1} triggered OOM on AMD RX 7600 XT. Downscale channels."
                    raise optuna.exceptions.TrialPruned("OOM crash - model too large")
                else:
                    raise e

            final_acc = results["final_val_accuracy"]
            print(f"🏆 Trial {trial.number + 1} Success! Student Val Accuracy: {final_acc:.2f}% | Params: {results['param_count']:,}")

            # 3. Save Every Trial's Weights (Requirement 2)
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

            if trial_progress_callback:
                trial_progress_callback({
                    "trial": trial.number + 1,
                    "model_name": class_name,
                    "accuracy": final_acc,
                    "params": results["param_count"],
                    "code_path": trial_checkpoint_path,
                    "code": code_str
                })

            return final_acc

        study.optimize(objective, n_trials=n_trials)

        # 4. Crown the Champion (Requirement 3)
        try:
            best_trial_id = study.best_trial.number
            best_checkpoint_path = f"{CHECKPOINT_DIR}/trial_{best_trial_id}.pt"
            champion_dest_path = "wonderland_champion.pt"

            if os.path.exists(best_checkpoint_path):
                shutil.copy(best_checkpoint_path, champion_dest_path)
                best_code_src = f"{CHECKPOINT_DIR}/trial_{best_trial_id}_model.py"
                if os.path.exists(best_code_src):
                    shutil.copy(best_code_src, "wonderland_champion.py")

                print(f"\n=======================================================")
                print(f"👑 ✨ [CHAMPION CROWNED] ✨ 👑")
                print(f"Winning Trial: Trial #{best_trial_id}")
                print(f"Best Validation Accuracy: {study.best_value:.2f}%")
                print(f"Saved Champion Weights: {os.path.abspath(champion_dest_path)}")
                print(f"Saved Champion Code:    {os.path.abspath('wonderland_champion.py')}")
                print(f"🚀 Champion model is saved and ready for deployment!")
                print(f"=======================================================\n")
        except Exception as e:
            print(f"Could not crown champion: {e}")

        return study

def main():
    parser = argparse.ArgumentParser(description="Gibby's Magical AI Wonderland: NAS Engine for AMD GPUs")
    parser.add_argument("--trials", type=int, default=6, help="Number of Optuna NAS trials")
    parser.add_argument("--epochs", type=int, default=4, help="Max epochs per trial")
    parser.add_argument("--task", type=str, default="fashionmnist", choices=["fashionmnist", "cifar10"])
    parser.add_argument("--llm", type=str, default="qwen2.5-coder:7b", help="Local LLM model name")
    args = parser.parse_args()

    engine = GibbyNASOrchestrator(
        task_type=args.task,
        max_epochs_per_trial=args.epochs,
        llm_model=args.llm
    )
    study = engine.run_nas_study(n_trials=args.trials)

if __name__ == "__main__":
    main()
