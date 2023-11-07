import torch,os,re
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
WEIGHTS_NAME = "pytorch_model.bin"
def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model
def get_parameter_dtype(parameter: Union[nn.Module, "ModuleUtilsMixin"]):

    last_dtype = None
    for t in parameter.parameters():
        last_dtype = t.dtype
        if t.is_floating_point():
            return t.dtype

    if last_dtype is not None:
        return last_dtype

    else:
        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        last_tuple = None
        for tuple in gen:
            last_tuple = tuple
            if tuple[1].is_floating_point():
                return tuple[1].dtype

        return last_tuple[1].dtype
def shard_checkpoint(state_dict: Dict[str, torch.Tensor], max_shard_size: Union[int, str] = "10GB"):

    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = []
    current_block = {}
    current_block_size = 0
    total_size = 0

    for key, weight in state_dict.items():
        weight_size = weight.numel() * dtype_byte_size(weight.dtype)

        if current_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append(current_block)
            current_block = {}
            current_block_size = 0

        current_block[key] = weight
        current_block_size += weight_size
        total_size += weight_size

    sharded_state_dicts.append(current_block)

    if len(sharded_state_dicts) == 1:
        return {WEIGHTS_NAME: sharded_state_dicts[0]}, None

    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = WEIGHTS_NAME.replace(".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin")
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index
def convert_file_size_to_int(size: Union[int, str]):

    if isinstance(size, int):
        return size
    if size.upper().endswith("GIB"):
        return int(size[:-3]) * (2**30)
    if size.upper().endswith("MIB"):
        return int(size[:-3]) * (2**20)
    if size.upper().endswith("KIB"):
        return int(size[:-3]) * (2**10)
    if size.upper().endswith("GB"):
        int_size = int(size[:-2]) * (10**9)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("MB"):
        int_size = int(size[:-2]) * (10**6)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("KB"):
        int_size = int(size[:-2]) * (10**3)
        return int_size // 8 if size.endswith("b") else int_size
    raise ValueError("`size` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'.")

def dtype_byte_size(dtype):

    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8