"""Cached Hugging Face causal language model; inference never uses a hosted API."""
import os
import threading
from functools import lru_cache
from pathlib import Path

DEFAULT_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
DEFAULT_REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
GENERATION_LOCK = threading.RLock()


def generation_settings():
    model = os.getenv('GENERATION_MODEL', DEFAULT_MODEL)
    return {'provider': 'huggingface-local', 'model': model,
            'revision': os.getenv('GENERATION_REVISION') or (DEFAULT_REVISION if model == DEFAULT_MODEL else None),
            'device': 'cpu'}


def generation_ready():
    from huggingface_hub import try_to_load_from_cache
    settings = generation_settings()
    local = Path(settings['model'])
    if local.is_dir():
        return (local / 'config.json').exists() and any(local.glob('*.safetensors'))
    return all(isinstance(try_to_load_from_cache(settings['model'], name, revision=settings['revision']), str)
               for name in ['config.json', 'tokenizer_config.json', 'model.safetensors'])


@lru_cache(maxsize=1)
def load_generator(name, revision):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    torch.set_num_threads(max(1, min(32, int(os.getenv('LOCAL_MODEL_THREADS', '4')))))
    try:
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=True,
                                                 trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision, local_files_only=True,
                                                   trust_remote_code=False, dtype=torch.float32)
        model.to('cpu').eval()
    except OSError as exc:
        raise ValueError('Local answer model is missing. Run python app.py --download-generator first.') from exc
    return tokenizer, model


def generate_answer(messages):
    import torch
    settings = generation_settings()
    with GENERATION_LOCK:
        tokenizer, model = load_generator(settings['model'], settings['revision'])
        max_new = max(32, min(1024, int(os.getenv('GENERATION_MAX_NEW_TOKENS', '256'))))
        budget = min(8192, model.config.max_position_embeddings - max_new)
        retained = [dict(message) for message in messages]
        while True:
            inputs = tokenizer.apply_chat_template(retained, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors='pt')
            length = inputs['input_ids'].shape[-1]
            if length <= budget:
                break
            if len(retained) > 2:
                # Drop old conversation before shortening current evidence.
                del retained[1:min(3, len(retained) - 1)]
                continue
            ids = tokenizer.encode(retained[-1]['content'], add_special_tokens=False)
            keep = len(ids) - (length - budget) - 16
            if keep < 64:
                raise ValueError('Question and instructions exceed the local generation context budget')
            retained[-1]['content'] = tokenizer.decode(ids[:keep], skip_special_tokens=True)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                    repetition_penalty=1.15, no_repeat_ngram_size=4,
                                    pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(output[0][length:], skip_special_tokens=True).strip()
        if not answer:
            raise ValueError('Local answer model returned no text; try a shorter question')
        return answer
