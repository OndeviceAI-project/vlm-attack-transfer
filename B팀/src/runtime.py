"""Shared B-team Qwen runtime; all image tensors use raw 0..255 RGB units."""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from case_config import parse_case, expected_census

ROOT = Path(os.environ.get('ONDEVICE_AI_ROOT', Path.home() / 'ondevice-ai')).resolve()
BTEAM = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get('ONDEVICE_AI_REPO', BTEAM.parent)).resolve()
sys.path.insert(0, str(REPO / '\uacf5\ud1b5'))
import settings
import quant_rtn
SPEC = settings.MODELS['qwen2vl']
# Long decoded answers can exceed Python CSV's default 128 KiB field limit.
csv.field_size_limit(10_000_000)
CSV_FIELDS = ['model', 'case', 'image_id', 'input', 'n_tokens', 'hit_cap', 'latency_s', 'text']

def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(temp, path)

def save_csv(path, rows, fields=CSV_FIELDS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)

def model_metadata():
    meta = json.loads((ROOT / 'model.json').read_text())
    if meta['model_id'] != SPEC['model_id']:
        raise ValueError('Model metadata does not match common settings')
    return meta

def load_processor():
    return AutoProcessor.from_pretrained(model_metadata()['snapshot_path'],
        local_files_only=True, **SPEC['processor_kwargs'])

def prompt_text(processor):
    messages = [{'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': settings.PROMPT}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)

def read_image(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        if image.size != (SPEC['img_size'], SPEC['img_size']):
            raise ValueError(f'Expected 448x448 prepared PNG: {path}')
        return image.copy()

def verify_fixed_pixels(iid, clean, noise, adv=None):
    """Reconstruct the common noise and check the actual saved attack pixels."""
    original = np.asarray(clean).astype(np.int16)
    noisy = np.asarray(noise).astype(np.int16)
    if noisy.shape != original.shape:
        raise ValueError(f'Noise image shape differs: {iid}')
    signs = np.random.default_rng([settings.NOISE_SEED, int(iid)]).choice([-1, 1], size=original.shape)
    expected = np.clip(original + settings.EPS * signs, 0, 255).astype(np.uint8)
    if not np.array_equal(noisy, expected):
        raise ValueError(f'Noise pixels do not match the fixed seed/pattern: {iid}')
    if adv is None:
        return None
    attacked = np.asarray(adv).astype(np.int16)
    if attacked.shape != original.shape:
        raise ValueError(f'Attack image shape differs: {iid}')
    linf = int(np.abs(attacked - original).max())
    if linf > settings.EPS:
        raise ValueError(f'Actual attack PNG exceeded epsilon: {iid}, linf={linf}')
    return linf

def raw_pixels(image, device='cpu'):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().to(device)

def to_image(pixels):
    arr = pixels.detach().clamp(0, 255).round().to('cpu').permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(arr)

def encode(processor, image):
    enc = processor(text=prompt_text(processor), images=[image], return_tensors='pt')
    if enc['image_grid_thw'].tolist() != [[1, 32, 32]]:
        raise ValueError('Unexpected image grid')
    return enc

def pixel_values(processor, pixels):
    ip = processor.image_processor
    normalized = ip.rescale_and_normalize(pixels.unsqueeze(0),
        do_rescale=ip.do_rescale, rescale_factor=ip.rescale_factor,
        do_normalize=ip.do_normalize, image_mean=ip.image_mean, image_std=ip.image_std)
    return ip.patchify(normalized, patch_size=ip.patch_size,
        merge_size=ip.merge_size, temporal_patch_size=ip.temporal_patch_size)[0].squeeze(0)

def check_preprocessing(processor, image, device='cuda'):
    reference = encode(processor, image)['pixel_values']
    cpu = pixel_values(processor, raw_pixels(image))
    diff_cpu = (cpu - reference).abs().max().item()
    direct = pixel_values(processor, raw_pixels(image, device)).to(torch.bfloat16)
    expected = reference.to(device=device, dtype=torch.bfloat16)
    diff_model = (direct - expected).abs().max().item()
    if cpu.shape != (1024, 1176) or diff_cpu != 0 or diff_model != 0:
        raise ValueError(f'Preprocessing mismatch: cpu={diff_cpu}, model={diff_model}')
    return {'cpu_float32_max_abs': diff_cpu, 'model_bfloat16_max_abs': diff_model}

def eos_ids(model):
    ids = model.generation_config.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids)
    if len(ids) < 2:
        raise ValueError('Expected the complete Qwen EOS list')
    return ids

def token_status(tokens, eos, cap=settings.MAX_NEW_TOKENS):
    ended = any(int(token) in eos for token in tokens)
    return dict(n_tokens=len(tokens), hit_cap=len(tokens) == cap and not ended, ended_eos=ended)

@torch.no_grad()
def generate(model, processor, image, overrides=None):
    enc = encode(processor, image).to('cuda')
    enc['pixel_values'] = enc['pixel_values'].to(torch.bfloat16)
    kwargs = {**settings.GEN_KWARGS, 'eos_token_id': eos_ids(model), **(overrides or {})}
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.generate(**enc, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens = output[0, enc['input_ids'].shape[1]:].tolist()
    result = token_status(tokens, eos_ids(model), kwargs['max_new_tokens'])
    result.update(latency_s=elapsed, text=processor.tokenizer.decode(tokens, skip_special_tokens=True))
    return result, tokens

def extend_answer(base, target):
    out = dict(base)
    if target.ndim != 1 or target.numel() == 0:
        raise ValueError('A nonempty one-dimensional target is required')
    suffix = target.unsqueeze(0)
    out['input_ids'] = torch.cat([base['input_ids'], suffix], dim=1)
    out['attention_mask'] = torch.cat([base['attention_mask'], torch.ones_like(suffix)], dim=1)
    for key in ['mm_token_type_ids', 'token_type_ids']:
        if key in base:
            out[key] = torch.cat([base[key], torch.zeros_like(suffix)], dim=1)
    return out

def protected_weights(model):
    """Keep the original bf16 embedding/head and their shared storage."""
    weights = [model.get_input_embeddings().weight, model.get_output_embeddings().weight]
    if any(weight.dtype != torch.bfloat16 for weight in weights):
        raise ValueError('Input embeddings and lm_head must remain bf16')
    if weights[0].data_ptr() != weights[1].data_ptr():
        raise ValueError('Expected the original Qwen2-VL-2B embedding/head weight tie')
    return tuple((id(weight), weight.data_ptr(), weight._version, str(weight.dtype))
                 for weight in weights)


def load_case(case, attack=False):
    spec = parse_case(case)
    if attack and case != 'ref':
        raise ValueError('Attack generation must use ref')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A bf16-capable CUDA GPU is required')
    kwargs = dict(dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    path = model_metadata()['snapshot_path']
    change = None
    protected_before = None
    if spec['method'] == 'bnb':
        bits = spec['bits']
        skip = [prefix for part, prefix in SPEC['parts'].items() if part not in spec['targets']]
        if bits == 8:
            cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
        else:
            cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=skip)
        model = AutoModelForImageTextToText.from_pretrained(path,
            quantization_config=cfg, device_map={'': 0}, **kwargs).eval()
    else:
        model = AutoModelForImageTextToText.from_pretrained(path, **kwargs).to('cuda').eval()
        protected_before = protected_weights(model)
        if spec['method'] == 'rtn':
            change = quant_rtn.apply_rtn(model, SPEC['parts'], targets=spec['targets'],
                bits=spec['bits'], group=settings.QUANT_GROUP)
    protected_after = protected_weights(model)
    if protected_before is not None and protected_before != protected_after:
        raise ValueError('Quantization changed the protected embedding/head parameters')
    count = quant_rtn.census(model, SPEC['parts'])
    if not all(total > 0 for _, total in count.values()):
        raise ValueError('Missing expected model components')
    quant_rtn.check_census(count, targets=spec['targets'], names=settings.PART_NAMES)
    if count != expected_census(case):
        raise ValueError(f'Model Linear census differs from the verified Qwen revision: {count}')
    if attack:
        model.requires_grad_(False)
    return model, dict(census=count, weight_change_pct=change,
        eos_token_ids=eos_ids(model), attention='sdpa', dtype='bfloat16',
        model_footprint_bytes=model.get_memory_footprint(),
        storage='simulated_bf16' if spec['method']=='rtn' else ('bnb_packed' if spec['method']=='bnb' else 'bf16'),
        quantization=spec, protected_weights=dict(embeddings_dtype='bfloat16',
            head_dtype='bfloat16', embedding_head_tied=True,
            unchanged_after_rtn=protected_before is not None))

def image_ids(split, limit=None):
    if split not in ('dev', 'eval'): raise ValueError('Only dev/eval splits are allowed')
    with (REPO / '\uacf5\ud1b5/image_list.csv').open() as stream:
        ids = [f"{int(row['coco_id']):012d}" for row in csv.DictReader(stream) if row['split'] == split]
    expected = settings.N_DEV if split == 'dev' else settings.N_EVAL
    if len(ids) != expected: raise ValueError('Unexpected split size')
    if limit is not None:
        if split != 'dev' or limit < 1 or limit > expected:
            raise ValueError('Subset checks are allowed on dev only')
        ids = ids[:limit]
    return ids

def input_path(split, kind, iid):
    return ROOT / 'data/qwen2vl' / split / kind / (iid + '.png')

def source_label(path):
    path = Path(path).resolve()
    try: return str(path.relative_to(REPO))
    except ValueError: return 'B팀/' + str(path.relative_to(BTEAM))

def environment():
    return dict(settings_version=settings.SETTINGS_VERSION, model=model_metadata(),
        gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
        packages={p: importlib.metadata.version(p) for p in
            ['torch', 'torchvision', 'transformers', 'bitsandbytes', 'accelerate', 'pillow', 'numpy']},
        repo_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        code_sha256={source_label(p): sha256(p) for p in
            list(Path(__file__).parent.glob('*.py')) + [REPO / '\uacf5\ud1b5/settings.py', REPO / '\uacf5\ud1b5/quant_rtn.py']})

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
