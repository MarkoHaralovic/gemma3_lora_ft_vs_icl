#
# - Time to First Token (TTFT): latency before the first output token is produced.
# - End-to-End Request Latency : how long it takes from submitting a query to receiving the full response
# - Time Per Output Token (TPOT): average generation speed in tokens per second. (also known as Inter-token Latency (ITL))
# - Token Generation Time (TGT): duration from first to last token.
# - Total Latency: TTFT + TGT.
#
import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ.pop("TORCH_LOGS", None)
os.environ.pop("TORCH_COMPILE_DEBUG", None)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# loosely based on code acquired from https://github.com/rumanxyz/llm-perf-benchmark
import unsloth
import torch
import time
import GPUtil
import numpy as np
import traceback
import threading
from transformers import TextIteratorStreamer
from typing import Optional, Dict, Any, List, Union
from unsloth.chat_templates import get_chat_template
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import os
import csv
from datetime import datetime
from transformers import BitsAndBytesConfig
import torch._dynamo as dynamo


class GPUMonitor:
    def __init__(
        self, monitoring_interval: float = 0.1, gpu_indices: Optional[List[int]] = None
    ):
        self.monitoring_interval = monitoring_interval
        self.gpu_indices = gpu_indices if gpu_indices is not None else [0]
        self._mem_samples = []
        self._util_samples = []
        self._is_monitoring = False
        self._monitoring_thread = None
        self._gpu_memory_usage = []
        self._gpu_utilization = []

    def start(self):
        self._is_monitoring = True
        self._mem_samples.clear()
        self._util_samples.clear()

        def monitor_gpu():
            while self._is_monitoring:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        selected = [g for g in gpus if g.id in self.gpu_indices]

                        mem_vals = [float(g.memoryUsed) for g in selected]  # MB
                        util_vals = [float(g.load) * 100.0 for g in selected]  # %

                        self._gpu_memory_usage.extend(mem_vals)
                        self._gpu_utilization.extend(util_vals)

                        self._mem_samples.append(max(mem_vals))
                        self._util_samples.append(max(util_vals))

                    time.sleep(self.monitoring_interval)
                except Exception as e:
                    print(f"GPU monitoring error: {e}")
                    break

        self._monitoring_thread = threading.Thread(target=monitor_gpu, daemon=True)
        self._monitoring_thread.start()

    def stop(self):
        self._is_monitoring = False
        if self._monitoring_thread:
            self._monitoring_thread.join()

    def peak_mem(self) -> float:
        return max(self._mem_samples) if self._mem_samples else 0.0

    def p90_mem(self) -> float:
        return float(np.percentile(self._mem_samples, 90)) if self._mem_samples else 0.0

    def peak_util(self) -> float:
        return max(self._util_samples) if self._util_samples else 0.0

    def p90_util(self) -> float:
        return (
            float(np.percentile(self._util_samples, 90)) if self._util_samples else 0.0
        )


def benchmark_single_prompt(
    model,
    tokenizer,
    input_prompt_text: str,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
    max_new_tokens: int = 100,
    do_sample=True,
    num_beams: int = None,
    c=None,
    device: Optional[str] = None,
    gpu_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:

    gpu_monitor = GPUMonitor(monitoring_interval=0.1, gpu_indices=gpu_indices)
    gpu_monitor.start()

    start_input_process = time.time()

    is_sharded = (
        hasattr(model, "hf_device_map")
        and len(getattr(model, "hf_device_map") or {}) > 1
    )
    messages = [{"role": "user", "content": input_prompt_text}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    inputs = {"input_ids": input_ids}

    if not is_sharded and device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    input_process_time = time.time() - start_input_process

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask", None),
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": 1,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_beams": None,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    if num_beams and num_beams > 1 and do_sample:
        generation_kwargs["do_sample"] = False

    # Streaming generation setup
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    generation_kwargs["streamer"] = streamer

    generation_start_time = time.time()
    first_token_time = None
    result_holder = {}
    generated_decoded_tokens = []

    @dynamo.disable
    def safe_generate(m, **kwargs):
        return m.generate(**kwargs)

    def _generate():
        result_holder["out"] = safe_generate(model, **generation_kwargs)

    generation_thread = threading.Thread(target=_generate, daemon=True)
    generation_thread.start()

    for token in streamer:
        if first_token_time is None:
            first_token_time = time.time() - generation_start_time
            first_token_start_time = time.time()
        generated_decoded_tokens.append(token)

    generation_thread.join()
    gpu_monitor.stop()

    if "error" in result_holder:
        print(f"Generation failed: {result_holder['error']}")
        print(result_holder.get("traceback", ""))
        return {}

    if "out" not in result_holder:
        print("No output generated")
        return {}

    total_generation_time = time.time() - generation_start_time

    output = result_holder["out"]

    total_len = len(generated_decoded_tokens)
    input_tokens = int(inputs["input_ids"].shape[1])
    output_tokens = total_len

    if output_tokens == 0:
        print("Warning: No tokens generated")
        return {}

    # Safe timing calculations
    if first_token_time is None:
        first_token_time = 0.001

    token_generation_time = time.time() - first_token_start_time
    decode_time = token_generation_time
    if decode_time <= 0:
        decode_time = 0.001

    ttft = first_token_time
    total_tps = (
        (input_tokens + output_tokens) / total_generation_time
        if total_generation_time > 0
        else 0
    )
    decode_tps = output_tokens / decode_time if decode_time > 0 else 0

    # GPU metrics
    peak_gpu_usage = gpu_monitor.peak_mem()
    p90_gpu_usage = gpu_monitor.p90_mem()
    peak_gpu_utilization = gpu_monitor.peak_util()
    p90_gpu_utilization = gpu_monitor.p90_util()

    benchmark_results = {
        "total_generation_time": total_generation_time,
        "time_to_first_token_seconds": ttft,
        "token_generation_time": decode_time,
        "time_per_output_token": 1 / decode_tps if decode_tps > 0 else 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "tokens_per_second": total_tps,
        "output_decode_tokens_per_second": decode_tps,
        "input_process_time_seconds": input_process_time,
        "e2e_latency": ttft + total_generation_time,
        "peak_gpu_memory_mb": peak_gpu_usage,
        "p90_gpu_memory_mb": p90_gpu_usage,
        "peak_gpu_utilization": peak_gpu_utilization,
        "p90_gpu_utilization": p90_gpu_utilization,
    }

    return benchmark_results


def benchmark_language_model(
    model,
    tokenizer,
    prompts: List[str],
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
    max_new_tokens: int = 100,
    do_sample=True,
    num_beams: int = None,
    device: Optional[str] = None,
    gpu_indices: Optional[List[int]] = None,
) -> Dict[str, Union[float, List[Dict[str, Any]]]]:
    """
    Benchmark a language model's performance across multiple prompts.
    """

    prompt_results = []
    for prompt in prompts:
        print(f"prompt : {prompt}")
        result = benchmark_single_prompt(
            model=model,
            tokenizer=tokenizer,
            input_prompt_text=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            c=None,
            device=device,
            gpu_indices=gpu_indices,
        )
        if result:
            prompt_results.append(result)

    if not prompt_results:
        return {}

    ttft_list = [result["time_to_first_token_seconds"] for result in prompt_results]
    tpot_list = [result["time_per_output_token"] for result in prompt_results]
    tgt_list = [result["total_generation_time"] for result in prompt_results]
    e2e_latency_list = [result["e2e_latency"] for result in prompt_results]
    decode_tps_list = [
        result["output_decode_tokens_per_second"] for result in prompt_results
    ]
    gpu_usage_list = [result["peak_gpu_memory_mb"] for result in prompt_results]
    gpu_util_list = [result["peak_gpu_utilization"] for result in prompt_results]

    # Aggregate metrics
    aggregate_results = {
        # Time to First Token (TTFT) metrics
        "p50_ttft_seconds": round(np.percentile(ttft_list, 50), 3),
        "p90_ttft_seconds": round(np.percentile(ttft_list, 90), 3),
        # Time per output token (TPOT) metrics
        "p50_tpot_seconds": round(np.percentile(tpot_list, 50), 3),
        "p90_tpot_seconds": round(np.percentile(tpot_list, 90), 3),
        # Total generation time (TGT) metrics
        "p50_tgt_seconds": round(np.percentile(tgt_list, 50), 3),
        "p90_tgt_seconds": round(np.percentile(tgt_list, 90), 3),
        # End to end latency (e2e latency) metrics
        "p50_e2elatency_seconds": round(np.percentile(e2e_latency_list, 50), 3),
        "p90_e2elatency_seconds": round(np.percentile(e2e_latency_list, 90), 3),
        # Output Decode Tokens Per Second metrics
        "p50_decode_tps": round(np.percentile(decode_tps_list, 50), 3),
        "p90_decode_tps": round(np.percentile(decode_tps_list, 90), 3),
        # GPU Memory Usage metrics
        "max_gpu_memory_mb": round(max(gpu_usage_list), 3),
        "p90_gpu_memory_mb": round(np.percentile(gpu_usage_list, 90), 3),
        # GPU Utilization metrics
        "max_gpu_utilization": round(max(gpu_util_list), 3),
        "p90_gpu_utilization": round(np.percentile(gpu_util_list, 90), 3),
    }

    return aggregate_results


def clean_gpu():
    import os

    os.system(
        """
    echo "Cleaning up vLLM and CUDA contexts"
    pkill -f "vllm" || true
    pkill -f "engine_core" || true
    pkill -f "torchrun" || true
    sleep 2
    fuser -k /dev/nvidia* || true
    """
    )


clean_gpu()


def get_sample_prompts(n_shot: int, task_type: str) -> list:
   if task_type == "classification":
      base_prompt = "Classify the following news article into one of these categories: World, Sports, Business, Science/Technology."
      examples = ""
      sample_examples = [
         (
               "The stock market closed higher today as investors showed optimism about the latest tech earnings.",
               "Business",
         ),
         (
               "The football team secured a 3-1 victory in the championship game last night.",
               "Sports",
         ),
         (
               "NASA announced a new mission to explore the outer reaches of our solar system.",
               "Science/Technology",
         ),
         (
               "World leaders met today to discuss climate change policies at the United Nations.",
               "World",
         ),
      ]

      for i in range(min(n_shot, len(sample_examples))):
         text, label = sample_examples[i]
         examples += f"\nExample {i+1}:\nText: {text}\nCategory: {label}\n"

      query = "\nText: The national basketball league announced the start of the new season.\nCategory:"
      return [base_prompt + examples + query]

   elif task_type == "question_answering":
      base_prompt = "Read the passage and answer the question. If the answer is not in the passage, say 'unanswerable'."
      examples = ""
      sample_examples = [
         (
               "Context: Machine learning is a subfield of artificial intelligence that focuses on enabling systems to learn from data without being explicitly programmed.",
               "What does machine learning focus on?",
               "enabling systems to learn from data without being explicitly programmed.",
         ),
         (
               "Context: The Eiffel Tower is located in Paris, France, and was completed in 1889.",
               "Where is the Statue of Liberty located?",
               "unanswerable",
         ),
         (
               "Context: Water boils at 100 degrees Celsius under standard atmospheric pressure.",
               "At what temperature does water boil?",
               "100 degrees Celsius.",
         ),
      ]

      for i in range(min(n_shot, len(sample_examples))):
         context, question, answer = sample_examples[i]
         examples += (
               f"\nExample {i+1}:\n{context}\nQuestion: {question}\nAnswer: {answer}\n"
         )

      query = (
         "\nContext: Artificial intelligence enables computers to perform tasks that typically require human intelligence, such as visual perception and language understanding.\n"
         "Question: What is artificial intelligence?\nAnswer:"
      )

      return [base_prompt + examples + query]


UNIQUE_KEY_FIELDS = [
   "model_id",
   "adapter_id",
   "task_type",
   "n_shot",
   "decoding_strategy",
]


def load_existing_combinations(csv_path: str) -> set:
   combos = set()
   if not os.path.exists(csv_path):
      return combos
   with open(csv_path, "r", newline="") as f:
      reader = csv.DictReader(f)
      for row in reader:
         combo = (
               row.get("model_id", "").strip(),
               row.get("adapter_id", "").strip(),
               row.get("task_type", "").strip(),
               row.get("n_shot", "").strip(),
               row.get("decoding_strategy", "").strip(),
         )
         combos.add(combo)
   return combos


def combo_tuple(model_id, adapter_id, task_type, n_shot, ds_name):
   return (str(model_id), str(adapter_id), str(task_type), str(n_shot), str(ds_name))


def any_missing_for(model_id, adapter_id, task_type):
   for n_shot in icl_variants["k_shot"]:
      for ds_name in icl_variants["decoding_strategy"].keys():
         if (
               combo_tuple(model_id, adapter_id, task_type, n_shot, ds_name)
               not in existing_combos
         ):
               return True
   return False


CSV_HEADERS = [
   "timestamp",
   "model_id",
   "adapter_id",
   "task_type",
   "n_shot",
   "decoding_strategy",
   "temperature",
   "top_p",
   "top_k",
   "do_sample",
   "num_beams",
   "max_gen_toks",
   "p50_ttft_seconds",
   "p90_ttft_seconds",
   "p50_tpot_seconds",
   "p90_tpot_seconds",
   "p50_tgt_seconds",
   "p90_tgt_seconds",
   "p50_e2elatency_seconds",
   "p90_e2elatency_seconds",
   "p50_decode_tps",
   "p90_decode_tps",
   "max_gpu_memory_mb",
   "p90_gpu_memory_mb",
   "max_gpu_utilization",
   "p90_gpu_utilization",
]


def initialize_csv(csv_path: str):
   if not os.path.exists(csv_path):
      os.makedirs(os.path.dirname(csv_path), exist_ok=True)
      with open(csv_path, "w", newline="") as f:
         writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
         writer.writeheader()


def append_results_to_csv(csv_path: str, row_data: Dict[str, Any]):
   flat_data = {k: v for k, v in row_data.items() if k != "metrics"}
   flat_data.update(row_data.get("metrics", {}))

   with open(csv_path, "a", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
      writer.writerow(flat_data)


def get_bnb_config(model_id):
   if model_id == "google/gemma-3-12b-it":
      return BitsAndBytesConfig(
         load_in_8bit=True,
         llm_int8_threshold=6.0,
         llm_int8_has_fp16_weight=False,
      )
   if model_id == "google/gemma-3-27b-it":
      return BitsAndBytesConfig(
         load_in_4bit=True,
         bnb_4bit_compute_dtype=torch.bfloat16,
         bnb_4bit_quant_type="nf4",
         bnb_4bit_use_double_quant=True,
      )
   return None


def load_adapter(base_model_id, adapter_id, gpu_id):
   cfg = PeftConfig.from_pretrained(adapter_id)
   base_id = cfg.base_model_name_or_path or base_model_id

   tok = AutoTokenizer.from_pretrained(base_id, use_fast=True, trust_remote_code=True)
   base = AutoModelForCausalLM.from_pretrained(
      base_id,
      torch_dtype=torch.bfloat16,
      device_map={"": gpu_id},
      trust_remote_code=True,
   )
   model = PeftModel.from_pretrained(base, adapter_id)
   model.eval()
   tok = get_chat_template(tok, chat_template="gemma3")

   return model, tok

gemma3_lora_adapters = {
    "google/gemma-3-270m-it" :  {
        "classification" : "Mhara/google_gemma-3-270m-it_ft_ag_news_v3",
        "question_answering" : "Mhara/google_gemma-3-270m-it_ft_squad_v2"
    },
    "google/gemma-3-1b-it" :  {
        "classification" : "Mhara/google_gemma-3-1b-it_ft_ag_news_v3",
        "question_answering" : "Mhara/google_gemma-3-1b-it_ft_squad_v2"
    },
    "google/gemma-3-12b-it" :  {
        "classification" : "Mhara/google_gemma-3-12b-it_ft_ag_news",
        "question_answering" : "Mhara/google_gemma-3-12b-it_ft_squad_v2"
    },
    "google/gemma-3-27b-it" :  {
        "classification" : "Mhara/google_gemma-3-27b-it_ft_ag_news",
        "question_answering" : "Mhara/google_gemma-3-27b-it_ft_squad_v2"
    },
    "google/gemma-3-4b-it": {
        "classification": "Mhara/google_gemma-3-4b-it_ft_ag_news",
        "question_answering": "Mhara/google_gemma-3-4b-it_ft_squad_v2",
    }
}

icl_variants = {
   "k_shot": [0, 5, 10, 25],
   "decoding_strategy": {
      "default": {
         "temperature": 1,
         "top_p": 0.95,
         "top_k": 64,
         "max_gen_toks": 125,
         "do_sample": True,
         "num_beams": None,
      },
      "greedy": {
         "temperature": 0,
         "top_p": None,
         "top_k": None,
         "do_sample": False,
         "max_gen_toks": 125,
         "num_beams": None,
      },
      "beam": {
         "num_beams": 5,
         "temperature": 0,
         "top_p": None,
         "top_k": None,
         "do_sample": False,
         "max_gen_toks": 125,
      },
      "top_p": {
         "do_sample": True,
         "top_p": 0.9,
         "top_k": None,
         "temperature": 0.7,
         "max_gen_toks": 125,
         "num_beams": None,
      },
   },
}

CUDA_DEVICE_IND = 0
device = f"cuda:{CUDA_DEVICE_IND}"
gpu_indices = [CUDA_DEVICE_IND]

csv_output_path = "results/benchmark_results.csv"
initialize_csv(csv_output_path)
existing_combos = load_existing_combinations(csv_output_path)

torch.cuda.empty_cache()
clean_gpu()

for model_id, adapters in gemma3_lora_adapters.items():
   print(f"model_id : {model_id}")
   if model_id in ["google/gemma-3-12b-it", "google/gemma-3-27b-it"]:
      bnb_cfg = get_bnb_config(model_id)
   else:
      bnb_cfg = None
   try:
      if any_missing_for(model_id, "base_model", "classification"):
         base_model = AutoModelForCausalLM.from_pretrained(
               model_id,
               torch_dtype=torch.bfloat16,
               quantization_config=bnb_cfg,
               device_map={"": CUDA_DEVICE_IND},
               trust_remote_code=True,
         )
         tokenizer = AutoTokenizer.from_pretrained(model_id)

         for n_shot in icl_variants["k_shot"]:
               for ds_name, ds_config in icl_variants["decoding_strategy"].items():
                  if (
                     combo_tuple(
                           model_id, "base_model", "classification", n_shot, ds_name
                     )
                     in existing_combos
                  ):
                     print(
                           f"Skip run : {(model_id, 'base_model', 'classification', n_shot, ds_name)}"
                     )
                     continue
                  print(f"\n  Base model - {n_shot}-shot, {ds_name} decoding")

                  prompts = get_sample_prompts(n_shot, "classification")

                  try:
                     result = benchmark_language_model(
                           base_model,
                           tokenizer,
                           prompts,
                           temperature=ds_config.get("temperature", 1.0),
                           top_p=ds_config.get("top_p", 0.95),
                           top_k=ds_config.get("top_k", 64),
                           max_new_tokens=ds_config.get("max_gen_toks", 125),
                           do_sample=ds_config.get("do_sample", True),
                           num_beams=ds_config.get("num_beams", None),
                           device=device,
                           gpu_indices=gpu_indices,
                     )

                     row_data = {
                           "timestamp": datetime.now().isoformat(),
                           "model_id": model_id,
                           "adapter_id": "base_model",
                           "task_type": "classification",
                           "n_shot": n_shot,
                           "decoding_strategy": ds_name,
                           "temperature": ds_config.get("temperature"),
                           "top_p": ds_config.get("top_p"),
                           "top_k": ds_config.get("top_k"),
                           "do_sample": ds_config.get("do_sample"),
                           "num_beams": ds_config.get("num_beams"),
                           "max_gen_toks": ds_config.get("max_gen_toks"),
                           "metrics": result,
                     }

                     append_results_to_csv(csv_output_path, row_data)
                     existing_combos.add(
                           combo_tuple(
                              model_id,
                              "base_model",
                              "classification",
                              n_shot,
                              ds_name,
                           )
                     )

                  except Exception as e:
                     print(e)
                     continue
         del base_model
         torch.cuda.empty_cache()
         clean_gpu()

   except Exception as e:
      print(e)
      continue

   for task_type, adapter_id in adapters.items():
      try:
         if not any_missing_for(model_id, adapter_id, task_type):
               print(
                  f"Skipping adapter {adapter_id} ({task_type}) — all combos already logged."
               )
               continue

         model, tokenizer = load_adapter(model_id, adapter_id, CUDA_DEVICE_IND)

         for n_shot in icl_variants["k_shot"]:
               for ds_name, ds_config in icl_variants["decoding_strategy"].items():

                  combo = combo_tuple(
                     model_id, adapter_id, task_type, n_shot, ds_name
                  )  # <-- fixed
                  if combo in existing_combos:
                     print(f"Skip run : {combo}")
                     continue

                  print(f"\n  {task_type} - {n_shot}-shot, {ds_name} decoding")

                  prompts = get_sample_prompts(n_shot, task_type)

                  try:
                     result = benchmark_language_model(
                           model,
                           tokenizer,
                           prompts,
                           temperature=ds_config.get("temperature", 1.0),
                           top_p=ds_config.get("top_p", 0.95),
                           top_k=ds_config.get("top_k", 64),
                           max_new_tokens=ds_config.get("max_gen_toks", 125),
                           do_sample=ds_config.get("do_sample", True),
                           num_beams=ds_config.get("num_beams"),
                           device=device,
                           gpu_indices=gpu_indices,
                     )

                     row_data = {
                           "timestamp": datetime.now().isoformat(),
                           "model_id": model_id,
                           "adapter_id": adapter_id,
                           "task_type": task_type,
                           "n_shot": n_shot,
                           "decoding_strategy": ds_name,
                           "temperature": ds_config.get("temperature"),
                           "top_p": ds_config.get("top_p"),
                           "top_k": ds_config.get("top_k"),
                           "do_sample": ds_config.get("do_sample"),
                           "num_beams": ds_config.get("num_beams"),
                           "max_gen_toks": ds_config.get("max_gen_toks"),
                           "metrics": result,
                     }

                     append_results_to_csv(csv_output_path, row_data)
                     existing_combos.add(combo)
                  except Exception as e:
                     print(e)
                     continue

         del model, tokenizer
         torch.cuda.empty_cache()
         clean_gpu()

      except Exception as e:
         print(e)
         continue
