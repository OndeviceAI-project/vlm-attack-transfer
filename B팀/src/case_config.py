"""Versioned component cases; importing this module never loads ML packages."""
import re

COMPONENTS = {'connector': 'C', 'language': 'L', 'vcl': 'VCL'}
BACKENDS = ('rtn', 'bnb', 'both')
DEFAULT_BACKEND = 'both'
RTN_BITS = (8, 6, 5, 4, 3)
BNB_BITS = (8, 4)
EXPECTED_LINEAR_COUNTS = {'V': 128, 'C': 2, 'L': 196, 'H': 1}
MODEL_ID = 'Qwen/Qwen2-VL-2B-Instruct'
MODEL_REVISION = '895c3a49bc3fa70a340399125c650a463535e71c'
PROTOCOLS = {
    'common_v1': {
        'cap': 512,
        'image_set': 'qwen2vl_attack_common_v1_i300_cap512_s01',
    },
    'extended_v1': {
        'cap': 5000,
        'image_set': 'qwen2vl_attack_extended_v1_i500_cap5000_tol20_s01',
    },
}


def parse_case(case):
    if case == 'ref':
        return {'method': 'ref', 'targets': '', 'bits': 16}
    match = re.fullmatch(r'(rtn|bnb)-(V|C|L|VCL)-(\d+)', case)
    if not match:
        raise ValueError(f'Unsupported case (H must stay bf16): {case}')
    method, targets, bits = match.groups()
    bits = int(bits)
    allowed = RTN_BITS if method == 'rtn' else BNB_BITS
    if bits not in allowed:
        raise ValueError(f'Unsupported bit width: {case}')
    return {'method': method, 'targets': targets, 'bits': bits}


def component_cases(component, backend=DEFAULT_BACKEND):
    if component not in COMPONENTS or backend not in BACKENDS:
        raise ValueError('Choose connector/language/vcl and rtn/bnb/both')
    target = COMPONENTS[component]
    cases = ['ref']
    if backend in ('rtn', 'both'):
        cases += [f'rtn-{target}-{bits}' for bits in RTN_BITS]
    if backend in ('bnb', 'both'):
        cases += [f'bnb-{target}-{bits}' for bits in BNB_BITS]
    return cases


def profile(component=None, backend=DEFAULT_BACKEND):
    if component is None:
        return None
    return {
        'version': 'components_v1', 'component': component,
        'targets': COMPONENTS[component], 'backend': backend,
        'cases': component_cases(component, backend),
        'group_size': 64, 'parameter_scope': 'Linear.weight',
        'activation_quantization': 'none' if backend=='rtn' else 'backend_defined: LLM.int8 dynamic; NF4 bf16 compute',
        'preserved': ['lm_head', 'input_embeddings', 'bias', 'normalization', 'patch_embed'],
        'preserve_embedding_head_tie': True,
    }


def run_name(component, protocol, backend=DEFAULT_BACKEND, repeat='r01'):
    if protocol not in PROTOCOLS or not re.fullmatch(r'r[0-9]{2,}', repeat):
        raise ValueError('Unknown protocol or invalid run number (use r01, r02, ...)')
    component_cases(component, backend)
    stem = protocol.removesuffix('_v1')
    return f'qwen2vl_{component}_{stem}_components_v1_{backend}_{repeat}'


def expected_census(case):
    targets = parse_case(case)['targets']
    return {part: [total if part in targets else 0, total]
            for part, total in EXPECTED_LINEAR_COUNTS.items()}
