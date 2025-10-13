import os
from unsloth import FastModel
import torch
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

max_seq_length = 2048
four_bit_quantization=False
eight_bit_quantization=False


models_to_finentune = {
    "google/gemma-3-4b-it": {
        "question_answering": {
            "r": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_alpha": 512,
            "lora_dropout": 0.05,
            "bias": "none",
            "use_gradient_checkpointing": True,
            "use_rslora": False,
            "loftq_config": None
        },
    },
    "google/gemma-3-12b-it": {
        "classification": {
            "r": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_alpha": 256,
            "lora_dropout": 0.1,
            "bias": "none",
            "use_gradient_checkpointing": True,
            "use_rslora": False,
            "loftq_config": None
        },
        "question_answering": {
            "r": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_alpha": 256,
            "lora_dropout": 0.1,
            "bias": "none",
            "use_gradient_checkpointing": True,
            "use_rslora": False,
            "loftq_config": None
        },
    },
    "google/gemma-3-27b-it": {
        "classification": {
            "r": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_alpha": 256,
            "lora_dropout": 0.1,
            "bias": "none",
            "use_gradient_checkpointing": True,
            "use_rslora": False,
            "loftq_config": None
        },
        "question_answering": {
            "r": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_alpha": 256,
            "lora_dropout": 0.1,
            "bias": "none",
            "use_gradient_checkpointing": True,
            "use_rslora": False,
            "loftq_config": None
        },
    }
}


models_configs = {
    "google/gemma-3-270m-it": SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        warmup_steps=10,
        max_steps=500,                
        learning_rate=5e-5,          
        logging_steps=100,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs/gemma-3-270m-it",
        report_to="none",
    ),
    "google/gemma-3-1b-it": SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        warmup_steps=10,
        max_steps=500,
        learning_rate=3e-5,
        logging_steps=100,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs/gemma-3-1b-it",
        report_to="none",
    ),
    "google/gemma-3-4b-it": SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=4,   
        gradient_accumulation_steps=2,   
        warmup_steps=50,                
        max_steps=1000,
        learning_rate=2e-5,
        logging_steps=100,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs/gemma-3-4b-it",
        report_to="none",
    ),
    "google/gemma-3-12b-it": SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=1200,
        learning_rate=1.5e-5,
        logging_steps=100,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs/gemma-3-12b-it",
        report_to="none",
    ),
    "google/gemma-3-27b-it": SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=1200,
        learning_rate=1e-5,
        logging_steps=100,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs/gemma-3-27b-it",
        report_to="none",
    ),
}

from datasets import load_dataset

def convert_to_chatml(example):
    if "question" in example and "answers" in example:
        if isinstance(example["answers"], dict):
            if "text" in example["answers"]:
                answer_text = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""
            elif "value" in example["answers"]:
                answer_text = example["answers"]["value"]
            else:
                answer_text = ""
        else:
            answer_text = str(example["answers"])

        user_msg = f"Answer the following question based on the context:\n\n{example.get('context', '')}\n\nQuestion: {example['question']}"
        system_prompt = "You are a knowledgeable assistant that answers factual questions."

    elif "text" in example and "label" in example:
        label = example.get("label", "")
        user_msg = f"Classify the sentiment or topic of the following text:\n\n{example['text']}\n\nLabel:"
        system_prompt = "You are a helpful assistant that performs text classification tasks."
        if isinstance(label, int) and "label_names" in example:
            answer_text = example["label_names"][label]
        else:
            answer_text = str(label)
    else:
        user_msg = example.get("question", example.get("text", ""))
        answer_text = str(example.get("answer", ""))
        system_prompt = "You are a helpful assistant."

    return {
        "conversations": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer_text},
        ]
    }


def formatting_prompts_func(examples):

    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")for convo in convos]
    return {"text": texts}


def prepare_dataset(dataset_name, train_samples=None):
    if train_samples is not None:
        dataset = load_dataset(*dataset_name, split=f"train[:{train_samples}]")
    else:
        dataset = load_dataset(dataset_name, split="train")
    dataset_chatml = dataset.map(convert_to_chatml)
    dataset_formatted = dataset_chatml.map(formatting_prompts_func, batched=True)
    return dataset_formatted


def get_dataset(task):

    if task == "classification":
        return prepare_dataset("ag_news"), "ag_news"
    elif task == "question_answering":
        return prepare_dataset("squad_v2"), "squad_v2"
    
import csv, os, time

os.makedirs("lora_results", exist_ok=True)
CSV_PATH = "lora_results/lora_finetune_metrics.csv"

def _append_row_csv(path, row: dict):
    file_exists = os.path.exists(path)
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:writer.writeheader()
        writer.writerow(row)
        
def clean_gpu():
    import os
    os.system("""
    echo "Cleaning up vLLM and CUDA contexts"
    pkill -f "vllm" || true
    pkill -f "engine_core" || true
    pkill -f "torchrun" || true
    sleep 2
    fuser -k /dev/nvidia* || true
    """)
clean_gpu()


for model_name, task_lora in models_to_finentune.items(): 

    for task, lora_parameters in task_lora.items():                
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=four_bit_quantization,                
            load_in_8bit=eight_bit_quantization,
            full_finetuning=False,
        )

        model = FastModel.get_peft_model(model, **lora_parameters)
        tokenizer = get_chat_template(tokenizer, chat_template="gemma3")
        dataset, dn = get_dataset(task=task)

        # trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,                             
            eval_dataset=None,
            args=models_configs[model_name],
        )

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        gpu_idx = torch.cuda.current_device() 
        gpu_stats = torch.cuda.get_device_properties(gpu_idx)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        trainer_stats = trainer.train()

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
        
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "task": task,
            "dataset": dn,  
            "run_dir": models_configs[model_name].output_dir,

            "r": lora_parameters.get("r"),
            "lora_alpha": lora_parameters.get("lora_alpha"),
            "lora_dropout": lora_parameters.get("lora_dropout"),
            "bias": lora_parameters.get("bias"),
            "target_modules": "|".join(lora_parameters.get("target_modules", [])),
            "use_gradient_checkpointing": lora_parameters.get("use_gradient_checkpointing"),
            "use_rslora": lora_parameters.get("use_rslora"),
            "loftq_config": bool(lora_parameters.get("loftq_config")),


            "max_seq_length": max_seq_length,
            "four_bit_quantization": bool(four_bit_quantization),
            "eight_bit_quantization": bool(eight_bit_quantization),

            "learning_rate": models_configs[model_name].learning_rate,
            "per_device_train_batch_size": models_configs[model_name].per_device_train_batch_size,
            "gradient_accumulation_steps": models_configs[model_name].gradient_accumulation_steps,
            "warmup_steps": models_configs[model_name].warmup_steps,
            "max_steps": models_configs[model_name].max_steps,
            "weight_decay": models_configs[model_name].weight_decay,
            "optim": models_configs[model_name].optim,
            "lr_scheduler_type": getattr(models_configs[model_name], "lr_scheduler_type", None),

            "train_runtime_sec": trainer_stats.metrics.get("train_runtime"),
            "train_samples_per_sec": trainer_stats.metrics.get("train_samples_per_second"),
            "train_steps_per_sec": trainer_stats.metrics.get("train_steps_per_second"),
            "global_step": trainer_stats.metrics.get("global_step"),
            "train_loss": trainer_stats.metrics.get("train_loss"),
            "epoch": trainer_stats.metrics.get("epoch"),

            "gpu_name": gpu_stats.name,
            "gpu_total_gb": max_memory,
            "gpu_reserved_start_gb": start_gpu_memory,
            "gpu_reserved_peak_gb": used_memory,
            "gpu_reserved_train_gb": used_memory_for_lora,
            "gpu_reserved_peak_pct": used_percentage,
            "gpu_reserved_train_pct": lora_percentage,
        }

        _append_row_csv(CSV_PATH, row)
        print(f"[metrics] appended row to {CSV_PATH}")

        run_id = f"{model_name.replace('/', '_')}_ft_{dn}"
        os.makedirs("models", exist_ok=True)                    
        model.save_pretrained(f"models/{run_id}")
        tokenizer.save_pretrained(f"models/{run_id}")

        hf_token = os.environ.get("HF_TOKEN")
        model.push_to_hub(f"Mhara/{run_id}", token=hf_token)
        tokenizer.push_to_hub(f"Mhara/{run_id}", token=hf_token)
        
        clean_gpu()