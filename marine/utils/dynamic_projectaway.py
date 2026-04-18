"""dynamic_projectaway.py
=======================
Dynamic-PROJECTAWAY: A context-aware, layer-specific hallucination mitigation module
for LLaVA-1.5-7B.  Designed as a drop-in replacement for MARINE's GuidanceLogits.

Four integrated mechanisms
--------------------------
1. **Conditioning Dilution Monitor** — Tracks the Visual Attention Ratio (VAR) proxy
   at each decoding step. If the model's attention to image tokens drops below a
   configurable threshold, the intervention pipeline is activated.

2. **HGAI — Layers 5-18** (Visual Information Enrichment)
   Heads Guided Attention Intervention: computes approx. per-head attention to image
   patches, identifies "consensus" positions where *all* heads agree, then adds a
   correction to the attention output that amplifies consensus patches and suppresses
   scattered ones.  Forces the model to lock onto a single coherent visual region.

3. **PROJECTAWAY — Layers 19-26** (Semantic Refinement)
   Per-layer orthogonal projection: extracts a hallucination direction v_l from each
   target layer using a single guidance (negative-prompt) forward pass, then during
   generation subtracts the component of the hidden state along that direction:
       h_new = h - (h · v_l) * v_l
   Using per-layer vectors captures the geometry of the hallucinated concept as it
   evolves through the transformer's latent space.

4. **Attention Sink Redistribution — Layers 19-26**
   Detects image-token "sinks" (patches receiving anomalously concentrated attention)
   and redistributes that mass uniformly to the remaining semantic patches.  The
   correction is injected additively into the attention output via the output proj.

Flash-Attention compatibility
------------------------------
All three attention-based mechanisms use the "Targeted Hook Trick": instead of
calling model.generate(..., output_attentions=True) — which disables FlashAttention
and materialises the full N×N matrix — we register forward_pre_hooks that capture
the hidden_states and past_key_value inputs, then manually compute:

    attn_img ≈ softmax(Q_last @ K_img.T / √d_head)

where K_img is the slice of the KV cache at image token positions.  This is O(n_img)
per head rather than O(seq_len²), and FlashAttention continues to handle the actual
token generation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

# ---------------------------------------------------------------------------
# LLaVA-1.5 constants
# ---------------------------------------------------------------------------
IMAGE_TOKEN_INDEX: int = 32000   # ID of the <image> placeholder token in HF LLaVA-1.5
LLAVA_NUM_PATCHES: int = 576     # 24 × 24 patches for 336×336 input, patch_size=14


class DynamicProjectAway(LogitsProcessor):
    """
    Dynamic-PROJECTAWAY hallucination mitigation for LLaVA-1.5-7B.

    Registers PyTorch forward hooks on specific transformer layers to perform
    real-time, layer-specific interventions during text generation — without
    touching logits or requiring a second full model forward pass per step.

    Parameters
    ----------
    model : LlavaForConditionalGeneration
        The loaded model (on CUDA).
    guidance_ids : Tensor [B, S_neg]
        Tokenised negative prompt describing the hallucinated object.
    guidance_images : Tensor [B, C, H, W]
        Pixel values for the guidance run (same image as the main input).
    guidance_attn_mask : Tensor [B, S_neg]
        Attention mask for guidance_ids.
    input_ids : Tensor [B, S]
        Tokenised positive prompt — used once to locate image token positions.
    hgai_layer_range : (int, int)
        Inclusive (start, end) layer indices for HGAI  (default 5–18).
    pa_layer_range : (int, int)
        Inclusive (start, end) layer indices for PROJECTAWAY + redistribution
        (default 19–26).
    dilution_threshold : float
        VAR proxy value below which intervention is activated (default 0.10).
    hgai_amplify_factor : float
        Multiplier applied to consensus image-patch attention (default 2.0).
    sink_threshold : float
        An image patch is a "sink" when its attention weight exceeds
        `sink_threshold × mean_image_attention` (default 2.0).
    guidance_strength : float
        Unused — kept so that call-sites can swap GuidanceLogits ↔
        DynamicProjectAway without changing keyword arguments.
    tokenizer : optional
        Unused — kept for API compatibility.
    """

    def __init__(
        self,
        model,
        guidance_ids: torch.Tensor,
        guidance_images: torch.Tensor,
        guidance_attn_mask: torch.Tensor,
        input_ids: torch.Tensor,
        hgai_layer_range: Tuple[int, int] = (5, 18),
        pa_layer_range: Tuple[int, int] = (19, 26),
        dilution_threshold: float = 0.10,
        hgai_amplify_factor: float = 2.0,
        sink_threshold: float = 2.0,
        guidance_strength: float = 0.7,  # kept for API compat
        tokenizer=None,                  # kept for API compat
    ):
        self.model            = model
        self.guidance_strength = guidance_strength  # unused but preserved

        # ---- layer ranges ----
        self.hgai_layers        = list(range(hgai_layer_range[0], hgai_layer_range[1] + 1))
        self.pa_layers          = list(range(pa_layer_range[0],   pa_layer_range[1]   + 1))
        self.all_target_layers  = sorted(set(self.hgai_layers + self.pa_layers))

        # ---- hyper-parameters ----
        self.dilution_threshold  = dilution_threshold
        self.hgai_amplify_factor = hgai_amplify_factor
        self.sink_threshold      = sink_threshold

        # ---- runtime state (shared between hooks and __call__) ----
        self.intervention_active = False
        self.step_count          = 0
        self._hooks: List        = []
        self._storage: Dict      = {
            "last_var":     1.0,  # VAR proxy from the most recent forward pass
            "pre_hidden":   {},   # {layer_idx: hidden_states tensor}  pre-hook capture
            "pre_pkv":      {},   # {layer_idx: past_key_value object} pre-hook capture
        }

        # ---- Step 1: locate image tokens in sequence ----
        self.image_token_positions: Optional[torch.Tensor] = (
            self._identify_image_token_positions(input_ids)
        )

        # ---- Step 2: per-layer PROJECTAWAY vectors from one guidance forward pass ----
        self.pa_vectors: Dict[int, torch.Tensor] = self._extract_projectaway_vectors(
            guidance_ids, guidance_images, guidance_attn_mask
        )

        # ---- Step 3: register all hooks ----
        self._install_hooks()

    # =========================================================================
    # Setup Helpers
    # =========================================================================

    def _get_llama_layers(self) -> torch.nn.ModuleList:
        """Return the list of LlamaDecoderLayer modules from the LLaVA backbone.

        Tries all known HF LLaVA and original LLaVA repo structures in order:
          1. model.model.language_model.layers          (HF: language_model = LlamaModel)
          2. model.model.language_model.model.layers    (HF: language_model = LlamaForCausalLM)
          3. model.model.layers                         (direct shortcut)
          4. model.language_model.model.layers          (original LLaVA repo)
        """
        m = self.model

        # Path 1 — HF llava-hf where language_model IS LlamaModel (most common)
        try:
            layers = m.model.language_model.layers
            if isinstance(layers, torch.nn.ModuleList):
                return layers
        except AttributeError:
            pass

        # Path 2 — HF llava-hf where language_model is LlamaForCausalLM
        try:
            layers = m.model.language_model.model.layers
            if isinstance(layers, torch.nn.ModuleList):
                return layers
        except AttributeError:
            pass

        # Path 3 — Direct shortcut
        try:
            layers = m.model.layers
            if isinstance(layers, torch.nn.ModuleList):
                return layers
        except AttributeError:
            pass

        # Path 4 — Original LLaVA repo
        try:
            layers = m.language_model.model.layers
            if isinstance(layers, torch.nn.ModuleList):
                return layers
        except AttributeError:
            pass

        sub = list(m.model._modules.keys()) if hasattr(m, "model") else "N/A"
        raise AttributeError(
            "[DynamicProjectAway] Cannot find transformer layers. "
            f"model.model submodules: {sub}"
        )



    def _identify_image_token_positions(
        self, input_ids: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Locate the 576 image-patch positions in the full input sequence.

        LLaVA-1.5 embeds a single IMAGE_TOKEN_INDEX placeholder in input_ids; the
        vision encoder + projector expand it to LLAVA_NUM_PATCHES (576) embeddings.
        We find the placeholder's position in row 0 (template is shared across batch)
        and return the range [img_start, img_start + 576).

        Returns None if the placeholder is absent (triggers a fallback warning).
        """
        ids          = input_ids[0]   # [seq_len]
        img_positions = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]

        if len(img_positions) == 0:
            print(
                "[DynamicProjectAway] WARNING: IMAGE_TOKEN_INDEX not found in "
                "input_ids. Falling back to positions 1-576. "
                "Verify that input_ids still contains the raw placeholder."
            )
            return torch.arange(1, LLAVA_NUM_PATCHES + 1, device=input_ids.device)

        img_start = img_positions[0].item()
        return torch.arange(
            img_start, img_start + LLAVA_NUM_PATCHES, device=input_ids.device
        )

    def _extract_projectaway_vectors(
        self,
        guidance_ids:        torch.Tensor,
        guidance_images:     torch.Tensor,
        guidance_attn_mask:  torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Run ONE forward pass of the negative (guidance) prompt and capture the
        mean hidden state at each PROJECTAWAY layer as the hallucination direction.

        The guidance prompt describes the hallucinated object
        (e.g. "Imagine a microwave is present in this image.").  The hidden state
        at semantic refinement layers encodes this concept's representation.
        Unit-normalising gives the erasure direction v_l for each layer l.

        Returns
        -------
        Dict[int, Tensor[hidden_dim]]  — one unit vector per PA layer.
        """
        captured:   Dict[int, torch.Tensor] = {}
        temp_hooks: List                    = []
        layers                              = self._get_llama_layers()

        def make_capture_hook(layer_idx: int):
            def hook(module, input_args, output):
                hs = output[0].detach()                      # Typically [B, S, D] or [N, D]
                # Reshape to [N, D] to safely average over all tokens regardless of padding structure
                mean_hs = hs.reshape(-1, hs.size(-1)).mean(dim=0)
                captured[layer_idx] = F.normalize(mean_hs, dim=0).clone()
            return hook

        for l in self.pa_layers:
            temp_hooks.append(layers[l].register_forward_hook(make_capture_hook(l)))

        try:
            with torch.inference_mode():
                self.model(
                    input_ids=guidance_ids,
                    pixel_values=guidance_images,
                    attention_mask=guidance_attn_mask,
                )
        finally:
            for h in temp_hooks:
                h.remove()

        missing = [l for l in self.pa_layers if l not in captured]
        if missing:
            print(f"[DynamicProjectAway] WARNING: v_l extraction failed for layers {missing}.")

        return captured

    # =========================================================================
    # Hook Management
    # =========================================================================

    def _install_hooks(self) -> None:
        """
        Register all forward hooks on target LlamaDecoderLayer / LlamaAttention
        modules.

        Per-layer hook architecture
        ---------------------------
        All target layers (5–26):
            • register_forward_pre_hook on self_attn  →  capture hidden_states
                                                          and past_key_value

        HGAI layers (5–18):
            • register_forward_hook on self_attn  →  HGAI consensus amplification

        PA layers (19–26):
            • register_forward_hook on self_attn  →  VAR update + sink redistribution
            • register_forward_hook on decoder-layer  →  PROJECTAWAY projection
        """
        layers = self._get_llama_layers()

        for l in self.all_target_layers:
            attn_mod  = layers[l].self_attn
            layer_mod = layers[l]

            # ---- pre-hook: capture inputs ----
            try:
                pre_h = attn_mod.register_forward_pre_hook(
                    self._make_attn_pre_hook(l), with_kwargs=True
                )
            except TypeError:
                # PyTorch < 2.0 does not support with_kwargs on pre-hooks.
                # Fall back to a positional-only version.
                pre_h = attn_mod.register_forward_pre_hook(
                    self._make_attn_pre_hook_legacy(l)
                )
            self._hooks.append(pre_h)

            # ---- HGAI post-hook on attention ----
            if l in self.hgai_layers:
                self._hooks.append(
                    attn_mod.register_forward_hook(self._make_hgai_hook(l))
                )

            # ---- sink redistribution + VAR update on attention ----
            if l in self.pa_layers:
                self._hooks.append(
                    attn_mod.register_forward_hook(self._make_sink_hook(l))
                )
                # ---- PROJECTAWAY on decoder layer output ----
                self._hooks.append(
                    layer_mod.register_forward_hook(self._make_pa_hook(l))
                )

    def cleanup(self) -> None:
        """Remove all registered hooks.  Call this after model.generate() finishes."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _remove_hooks(self) -> None:
        """Alias for cleanup(), called by __del__."""
        self.cleanup()

    @staticmethod
    def _extract_first(output) -> torch.Tensor:
        """
        Robustly extract the first (hidden_states) element from a hook output.

        In older HuggingFace (≤ 4.46) LlamaDecoderLayer returns a plain Python
        tuple: (hidden_states, [attn_weights], [past_key_value]).
        In newer HF that uses GradientCheckpointingLayer as the base class, the
        decoder layer may return just the hidden_states tensor directly.
        This helper handles both cases.
        """
        if isinstance(output, torch.Tensor):
            return output
        return output[0]

    @staticmethod
    def _rebuild_output(output, new_first: torch.Tensor):
        """
        Reconstruct a hook return value with `new_first` replacing element 0.

        Handles three possible output types produced by different HF versions:
          • bare Tensor   — new transformers GradientCheckpointingLayer unwrappers
          • plain tuple   — classic (hidden_states, attn_weights, past_key_value)
          • ModelOutput   — dataclass; convert to tuple then reconstruct
        """
        if isinstance(output, torch.Tensor):
            # Bare tensor path: just return the modified tensor directly.
            return new_first
        if isinstance(output, tuple):
            return (new_first,) + output[1:]
        # ModelOutput / other sequence: convert to plain tuple first.
        try:
            as_tuple = tuple(output)
            return (new_first,) + as_tuple[1:]
        except Exception:
            return (new_first,)

    # =========================================================================
    # Shared Computation: Targeted Hook Trick
    # =========================================================================

    def _get_head_config(self, module) -> Tuple[int, int, int]:
        """
        Safely retrieve number of heads and head dimension.
        Handles different transformers versions where these attributes
        might be on the module or on its config.
        """
        # Try finding config on module, else fallback to model config
        config = getattr(module, "config", getattr(self.model, "config", getattr(self.model, "language_model", self.model).config))
        
        nH = getattr(module, "num_heads", getattr(config, "num_attention_heads", None))
        if nH is None:
            raise AttributeError("[DynamicProjectAway] Could not find num_heads or num_attention_heads")
            
        nKV = getattr(module, "num_key_value_heads", getattr(config, "num_key_value_heads", nH))
        
        D = getattr(module, "head_dim", None)
        if D is None:
            hidden_size = getattr(config, "hidden_size", nH * 128)
            D = hidden_size // nH
            
        return nH, nKV, D


    def _compute_img_attention(
        self,
        module,
        layer_idx: int,
        hidden: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Approximate attention from the *last query token* to each image patch.

        Implements the "Targeted Hook Trick" recommended for Flash-Attention compat:

            attn_img ≈ softmax( Q_last @ K_img.T  /  √d_head )

        Q_last  — query for the most-recent token, computed via q_proj(hidden[:, -1:])
        K_img   — key vectors at image-patch positions, sliced from the KV cache.

        The softmax is computed over image positions only (not the full sequence),
        giving RELATIVE attention among image patches.  This is sufficient for:
            • detecting consensus regions (HGAI)
            • identifying sinks (redistribution)
            • tracking VAR proxy (dilution monitor)

        Returns
        -------
        Tensor [B, num_heads, 1, n_img] or None on failure.
        """
        if self.image_token_positions is None:
            return None

        pkv      = self._storage["pre_pkv"].get(layer_idx)
        bsz, seq_len, _ = hidden.shape
        nH, nKV, D = self._get_head_config(module)

        with torch.no_grad():
            # Query: last token only  →  [B, 1, nH·D]
            q_last = module.q_proj(hidden[:, -1:, :])
            q_last = q_last.view(bsz, 1, nH, D).transpose(1, 2)   # [B, nH, 1, D]

            # Key: from KV cache (image positions added during prefill) or from hidden
            k_cache = self._get_k_from_cache_or_hidden(
                module, pkv, layer_idx, hidden, bsz, seq_len, nKV, D
            )
            if k_cache is None:
                return None

            # Expand for Grouped Query Attention
            if nKV != nH:
                k_cache = k_cache.repeat_interleave(nH // nKV, dim=1)  # [B, nH, S, D]

            actual_seq = k_cache.shape[2]
            img_pos    = self.image_token_positions.clamp(max=actual_seq - 1).to(k_cache.device)
            k_img      = k_cache[:, :, img_pos, :]   # [B, nH, n_img, D]

            scale  = math.sqrt(D)
            scores = torch.matmul(q_last, k_img.transpose(-2, -1)) / scale  # [B, nH, 1, n_img]
            return torch.softmax(scores, dim=-1)

    def _get_k_from_cache_or_hidden(
        self,
        module,
        pkv,
        layer_idx: int,
        hidden: torch.Tensor,
        bsz: int,
        seq_len: int,
        nKV: int,
        D: int,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve the full-sequence Key tensor for this layer.

        During cached generation (step > 1):  slice from DynamicCache / legacy tuple.
        During prefill (step 1, no cache):     compute K via k_proj(hidden).
        """
        if pkv is not None:
            try:
                if hasattr(pkv, "key_cache"):
                    # Newer HF (≥ 4.38): DynamicCache
                    return pkv.key_cache[layer_idx]              # [B, nKV, S, D]
                else:
                    # Legacy: tuple of (key_tensor, value_tensor)
                    return pkv[0]                                # [B, nKV, S, D]
            except (AttributeError, IndexError, TypeError):
                return None

        # Prefill path: compute from current hidden (covers all seq positions)
        k = module.k_proj(hidden)                                # [B, S, nKV·D]
        return k.view(bsz, seq_len, nKV, D).transpose(1, 2)     # [B, nKV, S, D]

    def _get_v_from_cache_or_hidden(
        self,
        module,
        pkv,
        layer_idx: int,
        hidden: torch.Tensor,
        bsz: int,
        seq_len: int,
        nKV: int,
        D: int,
    ) -> Optional[torch.Tensor]:
        """Retrieve full-sequence Value tensor — mirrors _get_k_from_cache_or_hidden."""
        if pkv is not None:
            try:
                if hasattr(pkv, "value_cache"):
                    return pkv.value_cache[layer_idx]            # [B, nKV, S, D]
                else:
                    return pkv[1]
            except (AttributeError, IndexError, TypeError):
                return None

        v = module.v_proj(hidden)
        return v.view(bsz, seq_len, nKV, D).transpose(1, 2)

    # =========================================================================
    # Hook Factories
    # =========================================================================

    def _make_attn_pre_hook(self, layer_idx: int):
        """
        Pre-hook (with_kwargs=True) on LlamaAttention.forward.

        Captures hidden_states and past_key_value *before* the attention
        computation runs, storing them in self._storage for use by post-hooks.
        These captured tensors are detached from the computation graph — all
        intervention is on the OUTPUT side to avoid disrupting autograd.
        """
        storage = self._storage

        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is not None:
                storage["pre_hidden"][layer_idx] = hidden.detach()

            # past_key_value: try kwargs first, then positional (arg index 3)
            pkv = kwargs.get("past_key_value", None)
            if pkv is None and len(args) > 3:
                pkv = args[3]
            storage["pre_pkv"][layer_idx] = pkv

        return hook

    def _make_attn_pre_hook_legacy(self, layer_idx: int):
        """
        Fallback pre-hook for PyTorch < 2.0 (no with_kwargs support).
        Only captures positional hidden_states; past_key_value unavailable.
        HGAI / redistribution fall back to computing from hidden during prefill.
        """
        storage = self._storage

        def hook(module, args):
            if args:
                storage["pre_hidden"][layer_idx] = args[0].detach()
            storage["pre_pkv"][layer_idx] = None

        return hook

    # ------------------------------------------------------------------
    # HGAI Hook  (Layers 5–18)
    # ------------------------------------------------------------------

    def _make_hgai_hook(self, layer_idx: int):
        """
        Post-hook on LlamaAttention (layers 5–18).

        Heads Guided Attention Intervention
        ------------------------------------
        1. Compute approximate attention from ALL query positions to image patches
           using Q @ K_img (per head).
        2. Build a CONSENSUS mask: image positions where EVERY head is above its
           own per-head mean attention →  positions the model globally agrees on.
        3. Compute attention correction:
               correction = (w_amplified − w_original) @ V_img
           where w_amplified boosts consensus positions × hgai_amplify_factor and
           suppresses the rest (then renormalised to keep valid distribution).
        4. Project the correction through o_proj and add it to attn_output.

        This nudges the model to focus on a consistent visual region *before*
        semantic refinement (layers 19-26) processes the information.

        Note: HGAI is applied during PREFILL (seq_len > 1) when all image and text
        tokens are simultaneously available for head-comparison.  During cached
        single-token generation steps (seq_len == 1) it is skipped for efficiency.
        """
        storage  = self._storage
        amplify  = self.hgai_amplify_factor
        img_pos_ = self.image_token_positions

        def hook(module, args, output):
            if not self.intervention_active or img_pos_ is None:
                return output

            hidden = storage["pre_hidden"].get(layer_idx)
            if hidden is None:
                return output

            bsz, seq_len, _ = hidden.shape
            if seq_len <= 1:
                # Skip during cached generation; HGAI targets the prefill stage.
                return output

            attn_out = output[0]    # [B, S, hidden_dim]
            nH, nKV, D = self._get_head_config(module)
            pkv = storage["pre_pkv"].get(layer_idx)

            with torch.no_grad():
                # Full Q for all positions  →  [B, nH, S, D]
                q_all = module.q_proj(hidden).view(bsz, seq_len, nH, D).transpose(1, 2)

                k_cache = self._get_k_from_cache_or_hidden(
                    module, pkv, layer_idx, hidden, bsz, seq_len, nKV, D
                )
                v_cache = self._get_v_from_cache_or_hidden(
                    module, pkv, layer_idx, hidden, bsz, seq_len, nKV, D
                )
                if k_cache is None or v_cache is None:
                    return output

                if nKV != nH:
                    groups  = nH // nKV
                    k_cache = k_cache.repeat_interleave(groups, dim=1)
                    v_cache = v_cache.repeat_interleave(groups, dim=1)

                actual_seq = k_cache.shape[2]
                img_pos    = img_pos_.clamp(max=actual_seq - 1).to(k_cache.device)
                k_img      = k_cache[:, :, img_pos, :]   # [B, nH, n_img, D]
                v_img      = v_cache[:, :, img_pos, :]   # [B, nH, n_img, D]

                scale      = math.sqrt(D)
                # Attention from all query positions to image patches: [B, nH, S, n_img]
                scores_img = torch.matmul(q_all, k_img.transpose(-2, -1)) / scale
                w_img      = torch.softmax(scores_img, dim=-1)

                # --- Consensus mask ---
                # Mean head attention over batch and sequence dims: [nH, n_img]
                mean_per_head = w_img.mean(dim=[0, 2])
                head_mean     = mean_per_head.mean(dim=-1, keepdim=True)   # [nH, 1]
                above_mean    = mean_per_head > head_mean                  # [nH, n_img]
                consensus     = above_mean.all(dim=0)                      # [n_img] bool

                if not consensus.any():
                    return output

                # --- Amplify consensus, suppress non-consensus, renormalise ---
                w_amp = w_img.clone()
                w_amp[:, :, :, consensus]  = w_amp[:, :, :, consensus]  * amplify
                w_amp[:, :, :, ~consensus] = w_amp[:, :, :, ~consensus] / amplify
                w_amp = w_amp / w_amp.sum(dim=-1, keepdim=True).clamp(min=1e-8)

                # --- Additive correction via (Δw) @ V_img → o_proj ---
                weight_diff = w_amp - w_img                              # [B, nH, S, n_img]
                correction  = torch.matmul(weight_diff, v_img)          # [B, nH, S, D]
                correction  = (
                    correction.transpose(1, 2)
                    .contiguous()
                    .view(bsz, seq_len, nH * D)
                )
                correction  = module.o_proj(correction)                 # [B, S, hidden]

                return self._rebuild_output(output, attn_out + correction)

        return hook

    # ------------------------------------------------------------------
    # Sink Redistribution + VAR Hook  (Layers 19–26)
    # ------------------------------------------------------------------

    def _make_sink_hook(self, layer_idx: int):
        """
        Post-hook on LlamaAttention (layers 19–26).

        Two responsibilities
        --------------------
        A. **VAR update**: Compute the mean approximated attention to image tokens
           and store it in self._storage["last_var"].  This runs at every step
           (intervention_active or not) so the dilution monitor is always live.

        B. **Sink redistribution** (when intervention_active):
           - Identify image "sinks": patches where attention > sink_threshold × mean
           - Redistribute their attention mass uniformly to non-sink (semantic) patches
           - Apply correction:  (w_new − w_old) @ V_img  →  o_proj  →  add to attn_out

        Only the last token position in attn_out is modified (during generation steps
        the generated token is always the last element).
        """
        storage  = self._storage
        sink_thr = self.sink_threshold
        img_pos_ = self.image_token_positions

        def hook(module, args, output):
            if img_pos_ is None:
                return output

            hidden = storage["pre_hidden"].get(layer_idx)
            if hidden is None:
                return output

            attn_out = self._extract_first(output)    # [B, S, hidden_dim]
            bsz, seq_len, _ = hidden.shape
            nH, nKV, D = self._get_head_config(module)
            pkv = storage["pre_pkv"].get(layer_idx)

            # ---- A. Always update VAR proxy ----
            w_img = self._compute_img_attention(module, layer_idx, hidden)
            if w_img is not None:
                storage["last_var"] = w_img.mean().item()

            if not self.intervention_active or w_img is None:
                return output

            # ---- B. Sink redistribution ----
            with torch.no_grad():
                v_cache = self._get_v_from_cache_or_hidden(
                    module, pkv, layer_idx, hidden, bsz, seq_len, nKV, D
                )
                if v_cache is None:
                    return output

                if nKV != nH:
                    v_cache = v_cache.repeat_interleave(nH // nKV, dim=1)

                actual_seq = v_cache.shape[2]
                img_pos    = img_pos_.clamp(max=actual_seq - 1).to(v_cache.device)
                v_img      = v_cache[:, :, img_pos, :]   # [B, nH, n_img, D]

                # w_img: [B, nH, 1, n_img]  (from _compute_img_attention)
                mean_attn       = w_img.mean(dim=-1, keepdim=True)            # [B, nH, 1, 1]
                sink_mask       = w_img > sink_thr * mean_attn                # [B, nH, 1, n_img]
                total_sink_mass = (w_img * sink_mask.float()).sum(dim=-1, keepdim=True)

                if total_sink_mass.max().item() < 0.01:
                    return output   # no significant sinks; skip

                n_semantic = (~sink_mask).sum(dim=-1, keepdim=True).float().clamp(min=1.0)
                gain       = total_sink_mass / n_semantic              # [B, nH, 1, 1]

                w_new              = w_img.clone()
                w_new[sink_mask]   = 0.0
                w_new              = w_new + (~sink_mask).float() * gain

                # Correction delta: [B, nH, 1, D]
                correction = torch.matmul(w_new - w_img, v_img)
                correction = (
                    correction.transpose(1, 2)
                    .contiguous()
                    .view(bsz, 1, nH * D)
                )
                correction = module.o_proj(correction)                 # [B, 1, hidden]

                attn_out_new                = attn_out.clone()
                attn_out_new[:, -1:, :]    += correction
                return self._rebuild_output(output, attn_out_new)

        return hook

    # ------------------------------------------------------------------
    # PROJECTAWAY Hook  (Layers 19–26)
    # ------------------------------------------------------------------

    def _make_pa_hook(self, layer_idx: int):
        """
        Post-hook on LlamaDecoderLayer (layers 19–26).

        Per-Layer Orthogonal PROJECTAWAY
        ---------------------------------
        Subtracts the component of the hidden state along the hallucination
        direction v_l specific to this layer:

            h_new = h  −  (h · v_l) * v_l

        Using per-layer unit vectors (extracted independently for each layer
        during the guidance forward pass) handles the geometric evolution of
        the concept's representation through successive transformer blocks —
        a shared "mean vector" would be imprecise at any individual layer.

        This hook is a near-zero overhead operation: one dot product and one
        scaled subtraction per (batch, seq, hidden_dim) tensor.
        """
        v = self.pa_vectors.get(layer_idx)

        def hook(module, args, output):
            if not self.intervention_active or v is None:
                return output

            hs    = self._extract_first(output)                      # [B, S, D] or bare D
            v_dev = v.to(hs.device)                                  # [D]

            # h_new = h − (h · v) * v
            proj_coeff   = (hs @ v_dev).unsqueeze(-1)                # [B, S, 1] or [N, 1]
            hs_projected = hs - proj_coeff * v_dev                   # broadcasts cleanly

            return self._rebuild_output(output, hs_projected)

        return hook

    # =========================================================================
    # LogitsProcessor Interface
    # =========================================================================

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Called by model.generate() after each forward pass.

        Reads the VAR proxy stored by the sink hook during the just-completed
        forward pass and decides whether to activate the intervention pipeline
        for the *next* forward pass.  Logits are returned unchanged — all
        interventions operate in hidden-state space via the registered hooks.

        Transition messages are printed when the intervention state changes to
        aid in debugging and experimental monitoring.
        """
        self.step_count += 1
        var              = self._storage.get("last_var", 1.0)
        was_active       = self.intervention_active
        self.intervention_active = var < self.dilution_threshold

        if self.intervention_active and not was_active:
            print(
                f"[DynamicProjectAway] Step {self.step_count:03d}: "
                f"Conditioning dilution detected  (VAR={var:.4f} < "
                f"{self.dilution_threshold}).  Activating HGAI + PA + Redistribution."
            )
        elif not self.intervention_active and was_active:
            print(
                f"[DynamicProjectAway] Step {self.step_count:03d}: "
                f"VAR recovered  (VAR={var:.4f}).  Deactivating intervention."
            )

        # DPA does not modify logits — return unchanged.
        return logits

    # =========================================================================
    # Cleanup
    # =========================================================================

    def __del__(self) -> None:
        self._remove_hooks()
