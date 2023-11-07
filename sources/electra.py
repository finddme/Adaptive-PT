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
Results = namedtuple('Results', [
    'loss',
    'mlm_loss',
    'disc_loss',
    'gen_acc',
    'disc_acc',
    'disc_labels',
    'disc_predictions'
])

def log(t, eps=1e-9):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)

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

class HiddenLayerExtractor(nn.Module):
    def __init__(self, net, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = output

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def forward(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

# main electra class

class Electra(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        generator,
        discriminator,
        *,
        num_tokens = None,
        discr_dim = -1,
        discr_layer = -1,
        mask_prob = 0.15,
        replace_prob = 0.85,
        random_token_prob = 0.,
        mask_token_id = 4,
        pad_token_id = 0,
        mask_ignore_token_ids = [],
        disc_weight = 50.,
        gen_weight = 1.,
        temperature = 1.):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.config = config

        if discr_dim > 0:
            self.discriminator = nn.Sequential(
                HiddenLayerExtractor(discriminator, layer = discr_layer),
                nn.Linear(discr_dim, 1)
            )
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob

        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

        self.temperature = temperature

        self.disc_weight = disc_weight
        self.gen_weight = gen_weight


    def forward(self, input_ids, **kwargs):
        ff=self.discriminator(input_ids)
        replace_prob = prob_mask_like(input_ids, self.replace_prob)
        no_mask = mask_with_tokens(input_ids, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        mask_indices = torch.nonzero(mask, as_tuple=True)

        masked_input = input_ids.clone().detach()

        gen_labels = input_ids.masked_fill(~mask, self.pad_token_id)

        masking_mask = mask.clone()

        if self.random_token_prob > 0:
            assert self.num_tokens is not None, 'Number of tokens (num_tokens) must be passed to Electra for randomizing tokens during masked language modeling'

            random_token_prob = prob_mask_like(input_ids, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, input_ids.shape, device=input_ids.device)
            random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            masked_input = torch.where(random_token_prob, random_tokens, masked_input)

            masking_mask = masking_mask & ~random_token_prob

        masked_input = masked_input.masked_fill(masking_mask * replace_prob, self.mask_token_id)

        logits = self.generator(masked_input, **kwargs)

        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2),
            gen_labels,
            ignore_index = self.pad_token_id
        )

        sample_logits = logits[mask_indices]

        sampled = gumbel_sample(sample_logits, temperature = self.temperature)

        disc_input = input_ids.clone()
        disc_input[mask_indices] = sampled.detach()

        disc_labels = (input_ids != disc_input).float().detach()

        non_padded_indices = torch.nonzero(input_ids!= self.pad_token_id, as_tuple=True)

        disc_logits = self.discriminator(disc_input, **kwargs)
        disc_logits = disc_logits.reshape_as(disc_labels)

        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits[non_padded_indices],
            disc_labels[non_padded_indices]
        )

        with torch.no_grad():
            gen_predictions = torch.argmax(logits, dim=-1)
            disc_predictions = torch.round((torch.sign(disc_logits) + 1.0) * 0.5)
            gen_acc = (gen_labels[mask] == gen_predictions[mask]).float().mean()
            disc_acc = 0.5 * (disc_labels[mask] == disc_predictions[mask]).float().mean() + 0.5 * (disc_labels[~mask] == disc_predictions[~mask]).float().mean()

        return Results(self.gen_weight * mlm_loss + self.disc_weight * disc_loss, mlm_loss, disc_loss, gen_acc, disc_acc, disc_labels, disc_predictions)

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