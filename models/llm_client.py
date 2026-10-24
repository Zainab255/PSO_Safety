"""Backbone LLM client: text generation and gray-box hidden-state access.

Hidden-state extraction follows Section III-E of the paper:

    {H^(l)} = M(B, A)  with output_hidden_states=True under torch.no_grad()   (Eq. 6)
    t*_k    = sum_j A_kj - 1   (index of the last non-padding token)            (Eq. 7)
    e_k     = H^(L)[k, t*_k, :]

Eq. (7) assumes right padding, so the extraction batch is always right-padded.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class LLMClient:
    """Wraps a HuggingFace causal LM for generation and embedding extraction."""

    def __init__(self, backbone_cfg: Dict):
        self.key: str = backbone_cfg["key"]
        self.model_name: str = backbone_cfg["name_or_path"]
        self.hf_token: Optional[str] = os.environ.get(backbone_cfg["hf_token_env"]) or None
        self.dtype = _DTYPES[backbone_cfg["dtype"]]
        self.device_map = backbone_cfg["device_map"]
        self.max_input_tokens: int = backbone_cfg["max_input_tokens"]
        gen = backbone_cfg["generation"]
        self.max_new_tokens: int = gen["max_new_tokens"]
        self.temperature: float = gen["temperature"]
        self.use_chat_template: bool = gen["use_chat_template"]

        self.tokenizer = None
        self.model = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "LLMClient":
        logger.info("Loading tokenizer and model %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {"dtype": self.dtype, "token": self.hf_token}
        if self.device_map:
            kwargs["device_map"] = self.device_map
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if not self.device_map:
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self.model.device

    def synchronize(self) -> None:
        """Flush pending GPU work so wall-clock timings are exact."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Chat formatting
    # ------------------------------------------------------------------

    @property
    def has_chat_template(self) -> bool:
        return getattr(self.tokenizer, "chat_template", None) is not None

    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Render messages with the backbone's chat template (may raise if a role is unsupported)."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def _generation_text(self, prompt: str) -> Tuple[str, bool]:
        """Return (text, add_special_tokens) for a single-turn user prompt."""
        if self.use_chat_template and self.has_chat_template:
            return self.apply_chat_template([{"role": "user", "content": prompt}]), False
        return prompt, True

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Sample a completion for one user prompt; returns only the new text."""
        text, add_special = self._generation_text(prompt)
        return self.generate_from_text(text, add_special, temperature, max_new_tokens)

    def generate_from_text(
        self,
        text: str,
        add_special_tokens: bool,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Complete an already-rendered model input; temperature 0 means greedy."""
        inputs = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens,
            truncation=True, max_length=self.max_input_tokens,
        ).to(self.device)
        temperature = self.temperature if temperature is None else temperature
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Hidden-state extraction
    # ------------------------------------------------------------------

    def encode_hidden(
        self,
        texts: Sequence[str],
        pooling: str,
        layer: int,
        add_special_tokens: bool = True,
        pool_char_spans: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """One batched forward pass; returns pooled hidden states [N, d] (float32, CPU).

        pooling:
            ``last``    — last non-padding token (Eq. 7)
            ``mean``    — mean over all non-padding positions
            ``persona`` — mean over the tokens inside ``pool_char_spans``
                          (the persona instruction of each framed input)
        """
        self.tokenizer.padding_side = "right"
        need_offsets = pooling == "persona"
        enc = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.max_input_tokens,
            add_special_tokens=add_special_tokens, return_tensors="pt",
            return_offsets_mapping=need_offsets,
        )
        offsets = enc.pop("offset_mapping", None)
        enc = enc.to(self.device)

        with torch.no_grad():
            outputs = self.model(**enc, output_hidden_states=True)
        hidden = outputs.hidden_states[layer]                     # [N, n_max, d]
        mask = enc["attention_mask"]                              # [N, n_max]

        if pooling == "last":
            last_idx = mask.sum(dim=1) - 1                        # Eq. (7)
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
        elif pooling == "mean":
            m = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1)
        elif pooling == "persona":
            if pool_char_spans is None:
                raise ValueError("persona pooling needs pool_char_spans.")
            sel = torch.zeros_like(mask, dtype=torch.bool)
            for i, (start, end) in enumerate(pool_char_spans):
                s, e = offsets[i, :, 0], offsets[i, :, 1]
                sel[i] = ((s >= start) & (e <= end) & (e > s)).to(sel.device)
            sel = sel & mask.bool()
            if not bool(sel.any(dim=1).all()):
                raise ValueError("A persona span contained no tokens.")
            m = sel.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling '{pooling}' (expected last, mean or persona).")
        return pooled.float().cpu()

    def warmup(self, texts: Sequence[str], passes: int, layer: int) -> None:
        """Throwaway forward passes before latency measurements."""
        for _ in range(passes):
            self.encode_hidden(texts, pooling="last", layer=layer)
        self.synchronize()
