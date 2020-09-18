from typing import Dict, Optional
from copy import deepcopy
from functools import wraps

__all__ = ['register_model']

MIRROR_URL = 'https://ark-weights.s3.eu-central.stackpathstorage.com/'


def register_model(**configurations: Dict):
    def annotate_fn(model_fn):
        @wraps(model_fn)
        def wrapper(*args,
                    pretrained: str = None,
                    load_state_dict: bool = True,
                    state_dict_url: Optional[str] = None,
                    **kwargs):
            if pretrained is not None:
                try:
                    config = deepcopy(configurations[pretrained])
                except KeyError:
                    raise ValueError('pretrained model for {} does not exist'
                                     .format(pretrained))

                if state_dict_url is not None:
                    del config['state_dict']
                else:
                    state_dict_url = config.pop('state_dict')
                    state_dict_url = MIRROR_URL + state_dict_url
                config.update(kwargs)
                model = model_fn(*args, **config)
                if load_state_dict:
                    state_dict = load_state_dict_from_url(state_dict_url,
                                                          map_location='cpu',
                                                          check_hash=True)
                    model.load_state_dict(state_dict)
                return model
            else:
                return model_fn(*args, **kwargs)

        return wrapper

    return annotate_fn


def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
    r"""A modified version of `torch.hub.load_state_dict_from_url`, which handles the new 
    serialization protocol diferently. 
    See <https://github.com/pytorch/pytorch/issues/43106> for more information.
    """
    import os
    import sys
    import warnings
    import errno
    import torch
    from urllib.parse import urlparse
    from torch.hub import get_dir, download_url_to_file, HASH_REGEX

    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    return torch.load(cached_file, map_location=map_location)
