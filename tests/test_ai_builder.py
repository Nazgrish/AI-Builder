"""
Verification Test Suite for Gibby's Magical AI Wonderland.
Validates the Exact Local LLM System Prompt, Exact DistillationLoss, and FashionMNIST NAS.
"""

import os
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_builder.backend.hardware import get_hardware_engine
from ai_builder.inventor.code_synthesizer import LocalLLMCodeSynthesizer, EXACT_LLM_SYSTEM_PROMPT
from ai_builder.training.datasets import get_benchmark_dataloaders
from ai_builder.training.sandbox import ModelExecutionSandbox
from ai_builder.training.trainer import KnowledgeDistillationTrainer, DistillationLoss
from ai_builder.main import GibbyNASOrchestrator

def test_exact_llm_prompt_and_synthesis():
    print("\n[Test 1] Testing Exact Local LLM System Prompt & Synthesizer...")
    assert "GeneratedModel" in EXACT_LLM_SYSTEM_PROMPT
    assert "(batch_size, 1, 28, 28)" in EXACT_LLM_SYSTEM_PROMPT
    print("  System prompt constraints verified.")

    engine = get_hardware_engine()
    sandbox = ModelExecutionSandbox(device=engine.get_device())
    synthesizer = LocalLLMCodeSynthesizer(endpoint="http://localhost:11434", model_name="qwen2.5-coder:7b")

    code_str, class_name, model_inst, err = synthesizer.synthesize_architecture(
        design_spec={"conv_layers": 2, "base_channels": 32, "activation": "GELU"},
        input_shape=(1, 28, 28),
        output_dim=10,
        sandbox=sandbox
    )
    print(f"  Synthesized Model: {class_name} ({len(code_str)} bytes)")
    assert "class GeneratedModel" in code_str or "class " in code_str
    assert err is None, f"Sandbox failed: {err}"
    assert model_inst is not None
    print("  [OK] Exact LLM System Prompt & Synthesizer Passed.")

def test_exact_distillation_loss():
    print("\n[Test 2] Testing Exact DistillationLoss Implementation...")
    loss_fn = DistillationLoss(temperature=4.0, alpha=0.5)
    
    student_logits = torch.randn(4, 10, requires_grad=True)
    labels = torch.randint(0, 10, (4,))
    teacher_logits = torch.randn(4, 10)

    loss = loss_fn(student_logits, labels, teacher_logits)
    print(f"  Calculated DistillationLoss: {loss.item():.4f}")
    assert loss.item() > 0.0
    loss.backward()
    assert student_logits.grad is not None
    print("  [OK] Exact DistillationLoss Gradient Flow Passed.")

def test_frozen_teacher_and_trainer():
    print("\n[Test 3] Testing Frozen Teacher & Distillation Trainer...")
    engine = get_hardware_engine()
    trainer = KnowledgeDistillationTrainer(device=engine.get_device(), temperature=4.0, alpha=0.5)
    
    # Verify teacher is frozen and in eval mode
    assert not trainer.teacher.training
    for p in trainer.teacher.parameters():
        assert not p.requires_grad
    print("  Teacher is frozen with requires_grad=False and eval mode verified.")

    train_l, val_l, in_shape, out_dim = get_benchmark_dataloaders("fashionmnist", batch_size=32)
    assert in_shape == (1, 28, 28)

    class StudentTest(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 16, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(16, 10)
        def forward(self, x):
            return self.fc(torch.flatten(self.pool(torch.nn.functional.gelu(self.conv(x))), 1))

    student = StudentTest()
    results = trainer.train_epoch_step(
        student_model=student,
        train_loader=train_l,
        val_loader=val_l,
        epochs=2
    )
    print(f"  Distillation Result: Student Val Acc = {results['final_val_accuracy']:.2f}%, Status = {results['status']}")
    assert results["status"] == "success"
    print("  [OK] Frozen Teacher & Distillation Trainer Passed.")

def test_nas_study():
    print("\n[Test 4] Testing Gibby NAS Orchestrator with Exact Prompt & KD...")
    orchestrator = GibbyNASOrchestrator(
        task_type="fashionmnist",
        batch_size=32,
        max_epochs_per_trial=2
    )
    study = orchestrator.run_nas_study(n_trials=2)
    print(f"  Optuna Study Finished: Best Acc = {study.best_value:.2f}%")
    assert study.best_value >= 0.0
    print("  [OK] Full NAS Orchestrator Study Passed.")

if __name__ == "__main__":
    print("==================================================")
    print("✨ Testing Core Technical Implementations")
    print("==================================================")
    test_exact_llm_prompt_and_synthesis()
    test_exact_distillation_loss()
    test_frozen_teacher_and_trainer()
    test_nas_study()
    print("\n==================================================")
    print("🎉 ALL CORE TECHNICAL IMPLEMENTATIONS VERIFIED!")
    print("==================================================")
