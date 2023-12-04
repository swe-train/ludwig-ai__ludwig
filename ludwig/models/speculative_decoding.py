import logging
from typing import Optional, Tuple

import torch
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def multinomial_sample_one_no_sync(probs_sort):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(
    model: PreTrainedModel, input_tokens: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(input_tokens, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token(
    model: PreTrainedModel, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model: PreTrainedModel,
    current_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    callback=lambda _: _,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    for _ in range(num_new_tokens):
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False) if (
        #     torch.cuda.is_available() and model.device.type == "cuda"
        # ) else contextlib.nullcontext():
        next_token, next_prob = decode_one_token(model, current_token, input_pos, **sampling_kwargs)
        input_pos += 1
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        new_probs.append(next_prob.clone())
        current_token = next_token.view(1, -1)
    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


def speculative_decode(
    model: PreTrainedModel,
    draft_model: PreTrainedModel,
    current_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs,
) -> torch.Tensor:
    # Draft model inference sequentially
    device = current_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=device)
    draft_tokens, draft_probs = decode_n_tokens(
        draft_model, current_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs
    )
    draft_tokens = torch.cat(draft_tokens)
    draft_probs = torch.stack(draft_probs)

    # Parallel inference on target model
    target_logits = model_forward(
        model,
        torch.cat([current_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=device),
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)

    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, speculate_k, device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k] / p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0:
        # All draft tokens are accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        model_forward(draft_model, draft_tokens[-1].view(1, -1), orig_input_pos + speculate_k)
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])
