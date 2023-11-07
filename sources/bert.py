import math
from functools import reduce
from collections import namedtuple

import torch,os,re
from torch import nn
import torch.nn.functional as F

from transformers import ElectraForPreTraining, ElectraModel, ElectraTokenizer, ElectraForMaskedLM

from transformers import (
    DataCollatorForLanguageModeling,PretrainedConfig,
    ElectraForPreTraining, ElectraModel, ElectraTokenizer, ElectraForMaskedLM
    )
from sources.save_model import (
    dtype_byte_size, unwrap_model,
    get_parameter_dtype,shard_checkpoint,
    convert_file_size_to_int,
    WEIGHTS_NAME
    )

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import math
from functools import reduce

import torch
from torch import nn
import torch.nn.functional as F

def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

class BERT(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        transformer,
        mask_prob = 0.15,
        replace_prob = 0.9,
        num_tokens = None,
        random_token_prob = 0.,
        mask_token_id = 2,
        pad_token_id = 0,
        mask_ignore_token_ids = []):
        super().__init__()

        self.transformer = transformer
        self.config = config

        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob

        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

    def forward(self, seq, **kwargs):
        no_mask = mask_with_tokens(seq, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        masked_seq = seq.clone().detach()

        labels = seq.masked_fill(~mask, self.pad_token_id)

        if self.random_token_prob > 0:
            assert self.num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
            random_token_prob = prob_mask_like(seq, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, seq.shape, device=seq.device)
            random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)

            mask = mask & ~random_token_prob

        replace_prob = prob_mask_like(seq, self.replace_prob)
        masked_seq = masked_seq.masked_fill(mask * replace_prob, self.mask_token_id)

        logits = self.transformer(masked_seq, **kwargs)

        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index = self.pad_token_id
        )

        return mlm_loss
        
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "10GB",
        **kwargs,
    ):
        
        if "save_config" in kwargs:
            warnings.warn(
                "`save_config` is deprecated and will be removed in v5 of Transformers. Use `is_main_process` instead."
            )
            is_main_process = kwargs.pop("save_config")

        if os.path.isfile(save_directory):
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id, token = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        model_to_save = unwrap_model(self)

        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.torch_dtype = str(dtype).split(".")[1]

        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        if is_main_process:
            model_to_save.config.save_pretrained(save_directory)

        if state_dict is None:
            state_dict = model_to_save.state_dict()

        shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size)

        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            if (
                filename.startswith(WEIGHTS_NAME[:-4])
                and os.path.isfile(full_filename)
                and filename not in shards.keys()
                and is_main_process
            ):
                os.remove(full_filename)

        for shard_file, shard in shards.items():
            save_function(shard, os.path.join(save_directory, shard_file))

        if index is not None:
            save_index_file = os.path.join(save_directory, WEIGHTS_INDEX_NAME)
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)

        if push_to_hub:
            self._upload_modified_files(
                save_directory, repo_id, files_timestamps, commit_message=commit_message, token=token
            )

