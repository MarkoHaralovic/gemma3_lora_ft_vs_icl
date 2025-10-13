import unsloth
import os
import json
from peft import PeftModel, PeftConfig
from unsloth.chat_templates import get_chat_template 
import lm_eval
from lm_eval import evaluator, tasks
from lm_eval.utils import setup_logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp 
setup_logging("INFO") 

gemma3_270m_pt_args = (
    "pretrained=google/gemma-3-270m,"
    "tokenizer=google/gemma-3-270m,"
    "dtype=bfloat16,"
    "trust_remote_code=True"
)

gemma3_270m_it_args = (
    "pretrained=google/gemma-3-270m-it,"
    "tokenizer=google/gemma-3-270m-it,"
    "dtype=bfloat16,"
    "trust_remote_code=True"
)

gemma3_1b_pt_args = (
    "pretrained=google/gemma-3-1b-pt,"
    "tokenizer=google/gemma-3-1b-pt,"
    "dtype=bfloat16,"
    "trust_remote_code=True"
)

gemma3_1b_it_args = (
    "pretrained=google/gemma-3-1b-it,"
    "tokenizer=google/gemma-3-1b-it,"
    "dtype=bfloat16,"
 
)

gemma3_4b_pt_args = (
    "pretrained=google/gemma-3-4b-pt,"
    "tokenizer=google/gemma-3-4b-pt,"
    "dtype=bfloat16,"
    "trust_remote_code=True"
)

gemma3_4b_it_args = (
    "pretrained=google/gemma-3-4b-it,"
    "tokenizer=google/gemma-3-4b-it,"
    "dtype=bfloat16,"

)

gemma3_12b_pt_args = (
    "pretrained=google/gemma-3-12b-pt,"
    "tokenizer=google/gemma-3-12b-pt,"
    "dtype=bfloat16,"
    "trust_remote_code=True"
)

gemma3_12b_it_args = (
    "pretrained=google/gemma-3-12b-it,"
    "tokenizer=google/gemma-3-12b-it,"
    "dtype=bfloat16,"
    "trust_remote_code=True"
)

gemma3_27b_pt_args = (
    "pretrained=google/gemma-3-27b-pt,"
    "tokenizer=google/gemma-3-27b-pt,"
    "dtype=bfloat16,"
    "trust_remote_code=True"
)

gemma3_27b_it_args = (
    "pretrained=google/gemma-3-27b-it,"
    "tokenizer=google/gemma-3-27b-it,"
    "dtype=bfloat16," 
    "trust_remote_code=True"
)


gemma3_models = {
    "google/gemma-3-270m-pt" : gemma3_270m_pt_args,
    "google/gemma-3-270m-it" : gemma3_270m_it_args,
    "google/gemma-3-1b-pt" : gemma3_1b_pt_args,   
    "google/gemma-3-1b-it" : gemma3_1b_it_args,   
    "google/gemma-3-4b-pt" : gemma3_4b_pt_args,
    "google/gemma-3-4b-it" : gemma3_4b_it_args,
    "google/gemma-3-12b-pt" : gemma3_12b_pt_args,
    "google/gemma-3-12b-it" : gemma3_12b_it_args,
    "google/gemma-3-27b-pt" : gemma3_27b_pt_args,
    "google/gemma-3-27b-it" : gemma3_27b_it_args
}


# TODO add adapters evaluation
gemma3_lora_adapters  = {
    "google/gemma-3-270m-it" :  {
        "classification" : "Mhara/google_gemma-3-270m-it_ft_ag_news",
        "question_answering" : "Mhara/google_gemma-3-270m-it_ft_squad_v2"
    },
    "google/gemma-3-1b-it" :  {
        "classification" : "Mhara/google_gemma-3-1b-it_ft_ag_news",
        "question_answering" : "Mhara/google_gemma-3-1b-it_ft_squad_v2"
    },
    "google/gemma-3-4b-it" :  {
        "classification" : "Mhara/google_gemma-3-4b-it_ft_ag_news",
        "question_answering" : "Mhara/google_gemma-3-4b-it_ft_squad_v2"
    },
    "google/gemma-3-12b-it" :  {
        "classification" : "Mhara/google_gemma-3-12b-it_ft_ag_news",
        "question_answering" : "Mhara/google_gemma-3-12b-it_ft_squad_v2"
    },
    "google/gemma-3-27b-it" :  {
        "classification" : "Mhara/google_gemma-3-27b-it_ft_ag_news",
        "question_answering" : "Mhara/google_gemma-3-27b-it_ft_squad_v2"
    },
} 

def load_adapter(base_model_id, adapter_id):
    cfg = PeftConfig.from_pretrained(adapter_id)
    base_id = cfg.base_model_name_or_path or base_model_id

    _tok = AutoTokenizer.from_pretrained(base_id, use_fast=True, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    _model = PeftModel.from_pretrained(base, adapter_id)
    _model.eval()
    tok = get_chat_template(_tok, chat_template="gemma3")

    return _model, _tok


datasets_to_evaluate_on = {
    "question_answering"  : [
        "squadv2", # SQuAD 
        "triviaqa", 
        "nq_open", # natural queustions testing long context evaluation
        "boolq", #boolq
        "social_iqa", # social  QA 
    ],
    "classification" : [
        "ag_news", #AG news
        "sst2",
        "hellaswag", # HellaSwag
        "arc_easy",
        "piqa" #piqa
    ]
}

icl_variants = {
    "k_shot": [0, 5, 10, 25],
    "decoding_strategy": {
        "default": {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_gen_toks": 125,
            "do_sample": True
        },
        "greedy": {
            "temperature": 0,
            "do_sample": False,
            "max_gen_toks": 125
        },
        "beam": {
            "num_beams": 5,
            "temperature": 0,
            "do_sample": False,
            "max_gen_toks": 125
        },
        "top_p": {
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7,
            "max_gen_toks": 125
        }
    }
}

gemma3_models_evaluation_values = {
    "google/gemma-3-270m-pt" :{"n_shot" : 0, "ds_name"  : "default", "ds_kwargs" :  icl_variants["decoding_strategy"]["default"]},
    "google/gemma-3-270m-it" :{"n_shot" : 0, "ds_name"  : "default", "ds_kwargs" :  icl_variants["decoding_strategy"]["default"]},
    "google/gemma-3-1b-pt" :  {"n_shot" : 0, "ds_name"  : "default", "ds_kwargs" :  icl_variants["decoding_strategy"]["default"]}, 
    "google/gemma-3-1b-it" :  {"n_shot" : 0, "ds_name"  : "default", "ds_kwargs" :  icl_variants["decoding_strategy"]["default"]}, 
    "google/gemma-3-4b-pt" :  {"n_shot" : 0, "ds_name"  : "default", "ds_kwargs" :  icl_variants["decoding_strategy"]["default"]},
    "google/gemma-3-4b-it" :  {"n_shot" : 0, "ds_name"  : "default", "ds_kwargs" :  icl_variants["decoding_strategy"]["default"]},
    "google/gemma-3-12b-pt" : {"n_shot" : 0, "ds_name"  : "default", "ds_kwargs" :  icl_variants["decoding_strategy"]["default"]},
    "google/gemma-3-12b-it" : {"n_shot" : 0, "ds_name"  : "default", "ds_kwargs" :  icl_variants["decoding_strategy"]["default"]},
    "google/gemma-3-27b-pt" : {"n_shot" : 0, "ds_name"  : "default", "ds_kwargs" :  icl_variants["decoding_strategy"]["default"]},
    "google/gemma-3-27b-it" : {"n_shot" : 0, "ds_name"  : "default", "ds_kwargs" :  icl_variants["decoding_strategy"]["default"]},
}


def evaluate_single_run(model_name, model_id, task, n_shot, ds_name, ds_kwargs, model_args_str, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"\n🔹 [{ds_name}] {model_id} | task={task} | k={n_shot} | GPU={gpu_id}")
    results = evaluator.simple_evaluate(
        model=model_name,
        model_args=model_args_str,
        tasks=[task],
        num_fewshot=n_shot,
        gen_kwargs=ds_kwargs,
        batch_size="auto",
        device=f"cuda:{str(gpu_id)}",
    )["results"]

    out_dir = os.path.join("results", model_id.replace("/", "_"), task)
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.join(out_dir, f"{model_id.replace('/', '_')}_{n_shot}shot_{ds_name}.json")
    with open(out_name, "w") as f:
        json.dump(results[task], f, indent=2)
    print(f"Saved: {out_name}")

    res = results[task]
    metric = (
        res.get("acc_norm,none") or res.get("acc_norm") or res.get("acc") or
        res.get("em,none")       or res.get("em")       or
        res.get("f1,none")       or res.get("f1")       or float("-inf")
    )
    return (task, n_shot, ds_name, metric, out_name)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True) 
    gpu_pool = [1,2,3,4,6,7]
    model_name = "hf"
    jobs = []
    
    for model_id, model_args in gemma3_models.items():
        if not model_id.endswith("it"):
            continue
        base_args = "".join([p for p in model_args if p])

        for task_type, task_list in datasets_to_evaluate_on.items():
            adapter_map = gemma3_lora_adapters.get(model_id, {})
            adapter_id = adapter_map.get("classification" if task_type=="classification" else "qa")

            for task in task_list:
                ds_name = gemma3_models_evaluation_values[model_id]["ds_name"]
                ds_kwargs = gemma3_models_evaluation_values[model_id]["ds_kwargs"]
                n_shot  =  gemma3_models_evaluation_values[model_id]["n_shot"]
                gpu_id = gpu_pool[len(jobs) % len(gpu_pool)]
                model_args_str = base_args + f",peft={adapter_id}"

                jobs.append((model_name, model_id, task, n_shot, ds_name, ds_kwargs, model_args_str, gpu_id))
                
    results_lora = []
    with ProcessPoolExecutor(max_workers=len(gpu_pool)) as ex:
        futures = [ex.submit(evaluate_single_run, *args) for args in jobs]
        for fut in as_completed(futures):
            results_lora.append(fut.result())
    