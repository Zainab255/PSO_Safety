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
        self.batch_size: int = int(gen.get("batch_size", 8))
        self.top_p: float = float(gen.get("top_p", 1.0))
        self.top_k: int = int(gen.get("top_k", 0))
        self.chat_template_file: Optional[str] = backbone_cfg.get("chat_template_file")

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
        if self.chat_template_file and not getattr(self.tokenizer, "chat_template", None):
            from config import resolve_path
            self.tokenizer.chat_template = resolve_path(self.chat_template_file).read_text()
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
            "remove_invalid_values": True,
        }
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=self.top_p, top_k=self.top_k)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_batch(
        self,
        prompts: Sequence[str],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        rendered: bool = False,
    ) -> List[str]:
        """Sample completions for many prompts with left-padded batches (same sampling as ``generate``).

        Prompts are processed in length-sorted batches to limit padding; outputs keep input order.
        ``rendered=True`` means the prompts are already full model inputs (no chat template applied).
        """
        if not prompts:
            return []
        if rendered:
            texts, add_special = list(prompts), True
        else:
            pairs = [self._generation_text(p) for p in prompts]
            texts, add_special = [t for t, _ in pairs], pairs[0][1]
        temperature = self.temperature if temperature is None else temperature
        max_new = max_new_tokens or self.max_new_tokens
        bs = batch_size or self.batch_size
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        out: List[str] = [""] * len(texts)
        self.tokenizer.padding_side = "left"
        try:
            start = 0
            while start < len(order):
                idx = order[start:start + bs]
                try:
                    self._generate_chunk(texts, idx, out, add_special, temperature, max_new)
                except torch.cuda.OutOfMemoryError:
                    if bs == 1:
                        raise
                    torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                    logger.warning("CUDA OOM in batched generation; retrying with batch size %d", bs)
                    continue
                start += len(idx)
        finally:
            self.tokenizer.padding_side = "right"
        return out

    def _generate_chunk(self, texts, idx, out, add_special, temperature, max_new) -> None:
        enc = self.tokenizer(
            [texts[i] for i in idx], return_tensors="pt", padding=True,
            add_special_tokens=add_special, truncation=True, max_length=self.max_input_tokens,
        ).to(self.device)
        gen_kwargs = {"max_new_tokens": max_new, "pad_token_id": self.tokenizer.pad_token_id,
                      "do_sample": temperature > 0,
                      # fp16 Llama-2 can emit inf/nan logits; sanitise them instead of crashing CUDA.
                      "remove_invalid_values": True}
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=self.top_p, top_k=self.top_k)
        with torch.no_grad():
            ids = self.model.generate(**enc, **gen_kwargs)
        new = ids[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(idx):
            out[i] = self.tokenizer.decode(new[j], skip_special_tokens=True)

    def encode_all_layers(self, texts: Sequence[str], add_special_tokens: bool = True) -> torch.Tensor:
        """Last-token hidden state at every layer (index 0 = input embeddings): [N, L+1, d], float32 CPU."""
        self.tokenizer.padding_side = "right"
        enc = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.max_input_tokens,
            add_special_tokens=add_special_tokens, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.base_model(**enc, output_hidden_states=True)  # no LM head: hidden states only
        last_idx = (enc["attention_mask"].sum(dim=1) - 1).cpu()
        rows = torch.arange(last_idx.size(0))
        return torch.stack([h[rows.to(h.device), last_idx.to(h.device)].float().cpu()
                            for h in outputs.hidden_states], dim=1)

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
            outputs = self.model.base_model(**enc, output_hidden_states=True)  # no LM head: hidden states only
        hidden = outputs.hidden_states[layer]                     # [N, n_max, d]
        # With device_map="auto" layers live on different GPUs: index on the hidden state's device.
        mask = enc["attention_mask"].to(hidden.device)            # [N, n_max]

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
