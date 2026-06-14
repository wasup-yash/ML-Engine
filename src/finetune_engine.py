import os
from typing import Any, Dict, List
from uuid import uuid4

from src.logger import get_logger

logger = get_logger(__name__)


def _build_training_text(record: Dict[str, Any]) -> str:
    if "text" in record:
        text = str(record["text"])
        if "label" in record:
            return f"{text}\nLabel: {record['label']}"
        if "target" in record:
            return f"{text}\nTarget: {record['target']}"
        return text

    if "prompt" in record and "completion" in record:
        return f"{record['prompt']}\n{record['completion']}"

    raise ValueError(
        "Each dataset item must contain either: "
        "('text' and optional 'label'/'target') or ('prompt' and 'completion')."
    )


def run_lora_finetune(model: Any, dataset: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
    except ImportError as e:
        raise ImportError("finetune dependencies missing: transformers/datasets/peft") from e

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("dataset must be a non-empty list of labeled examples")

    if not hasattr(model, "config"):
        raise ValueError("LoRA fine-tuning currently supports HuggingFace-style torch models")

    base_model_id = config.get("tokenizer_path") or getattr(model.config, "_name_or_path", None)
    if not base_model_id:
        raise ValueError("Could not determine tokenizer path; set `tokenizer_path` in config")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))

    training_texts = [_build_training_text(item) for item in dataset]
    hf_dataset = Dataset.from_dict({"text": training_texts})
    max_length = int(config.get("finetune_max_length", 512))

    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized

    tokenized_dataset = hf_dataset.map(_tokenize, batched=True, remove_columns=["text"])

    lora_config = LoraConfig(
        r=int(config.get("lora_r", 8)),
        lora_alpha=int(config.get("lora_alpha", 16)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.get("lora_target_modules"),
    )
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)
    peft_model = get_peft_model(model, lora_config)

    output_root = config.get("adapter_output_dir", "./adapters")
    output_dir = os.path.join(output_root, f"adapter_{uuid4().hex}")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(config.get("finetune_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("finetune_grad_accum_steps", 4)),
        learning_rate=float(config.get("finetune_learning_rate", 2e-4)),
        num_train_epochs=float(config.get("finetune_num_epochs", 1.0)),
        logging_steps=int(config.get("finetune_logging_steps", 10)),
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    logger.info("Starting LoRA fine-tune job")
    trainer.train()
    peft_model.save_pretrained(output_dir, safe_serialization=True)
    logger.info(f"Saved adapter to {output_dir}")

    return {
        "model": peft_model,
        "adapter_path": output_dir,
    }
