"""Convert the pinned upstream checkpoint to the app's tensor-only model file."""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import typing

import torch

SOURCE_SHA256 = '39f5c14af0aa4140695781c8c52f13add3836da271b61a8fd6ade9fd4fddc5ff'
OUTPUT_SHA256 = '7a683ea84fee34ada68b1231b8ac81b3350a926ed2f2a1b6cbd3b9e503da880c'


class MetadataOnly:
    """Inert replacement; upstream configuration classes are never imported."""
    pass


def convert(source, destination):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('This is not the verified upstream best.ckpt checkpoint.')
    known = {'builtins.dict': dict, 'builtins.list': list, 'builtins.int': int,
             'typing.Any': typing.Any, 'collections.defaultdict': defaultdict}
    metadata = {'omegaconf.nodes.AnyNode', 'omegaconf.base.ContainerMetadata',
                'omegaconf.base.Metadata', 'omegaconf.dictconfig.DictConfig',
                'omegaconf.listconfig.ListConfig'}
    allowed = []
    for name in torch.serialization.get_unsafe_globals_in_checkpoint(source):
        if name in known:
            allowed.append((known[name], name))
        elif name in metadata:
            allowed.append((MetadataOnly, name))
        else:
            raise ValueError(f'Unexpected checkpoint type: {name}')
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    weights = {k.removeprefix('net.'): v for k, v in checkpoint['state_dict'].items() if k.startswith('net.')}
    if not weights or not all(isinstance(v, torch.Tensor) for v in weights.values()):
        raise ValueError('Missing model tensors.')
    if destination.name != 'neural-preset.pth':
        raise ValueError('The output filename must be neural-preset.pth for reproducible serialization.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Reserve exclusively so an existing installed model is never overwritten.
    with destination.open('xb'):
        pass
    try:
        torch.save(weights, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != OUTPUT_SHA256:
            raise ValueError('Converted checksum differs. Use the documented PyTorch 2.7.1 environment.')
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    print(f'Created {destination} (SHA-256: {digest})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=Path('models/neural-preset.pth'))
    args = parser.parse_args()
    convert(args.checkpoint, args.output)
