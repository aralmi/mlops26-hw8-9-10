"""Interchangeable backends. Importing this module never loads model weights."""
import time


class StubBackend:
    synthetic = True

    def __init__(self, manifest):
        self.loaded_id = manifest['model_id']
        self.runtime_sha256 = manifest['runtime_sha256']
        self.calls = 0

    def generate(self, prompt, limit):
        self.calls += 1
        time.sleep(.01)
        return {'answer': f'STUB ONLY: {self.loaded_id}', 'output_tokens': 1}


class TransformersBackend:
    synthetic = False

    def __init__(self, manifest):
        import importlib.metadata
        import random
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from journal import artifact_root
        runtime = manifest['runtime']
        for package, version in runtime['packages'].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f'Runtime package mismatch: {package}')
        if runtime['device'] != 'cpu' or runtime['dtype'] != 'float32':
            raise ValueError('This evaluated backend is pinned to CPU / float32')
        torch.set_num_threads(runtime['threads'])
        random.seed(runtime['generation']['seed'])
        np.random.seed(runtime['generation']['seed'])
        torch.manual_seed(runtime['generation']['seed'])
        root = artifact_root('.', manifest)
        tokenizer_root = root / manifest.get('tokenizer_subdir', '.')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
        base_root = root / manifest.get('base_subdir', '.')
        self.model = AutoModelForCausalLM.from_pretrained(base_root, dtype=torch.float32,
                                                        local_files_only=True)
        if manifest['model_kind'] == 'adapter':
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, root / manifest['adapter_subdir'],
                                                  local_files_only=True)
        self.model = self.model.to('cpu').eval()
        self.runtime = runtime
        self.loaded_id = manifest['model_id']
        self.runtime_sha256 = manifest['runtime_sha256']
        self.calls = 0

    def generate(self, prompt, limit):
        import torch
        g = self.runtime['generation']
        if limit != g['max_new_tokens']:
            raise ValueError('Generation limit differs from evaluated runtime')
        text = self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': g['system']}, {'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=g['enable_thinking'])
        batch = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        if batch['input_ids'].shape[1] > self.runtime['max_input_tokens']:
            raise ValueError('Input exceeds token limit')
        self.calls += 1
        with torch.inference_mode():
            output = self.model.generate(**batch, max_new_tokens=limit, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        tokens = output[0, batch['input_ids'].shape[1]:]
        return {'answer': self.tokenizer.decode(tokens, skip_special_tokens=True),
                'output_tokens': int(tokens.shape[0])}
