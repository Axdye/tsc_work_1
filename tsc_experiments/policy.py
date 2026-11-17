"""Exact sampled token retention; no regeneration between scoring and updating."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import re

from .common import PHASES


def parse_action(text):
    tags = re.findall(r"<signal>(.*?)</signal>", text, flags=re.I | re.S)
    if len(tags) != 1:
        return None
    label = tags[0].strip().upper()
    return PHASES.index(label) if label in PHASES else None


@dataclass
class Response:
    messages: list
    prompt_ids: list
    completion_ids: list
    text: str
    action: int | None
    truncated: bool = False
    advantage: float = 0.0
    reward: float = 0.0
    metadata: dict = field(default_factory=dict)


class ScriptedPolicy:
    """CPU infrastructure check ONLY. Not an LLM or a learned baseline."""

    def __init__(self, seed=0):
        import random

        self.random = random.Random(seed)
        self.calls = self.generated_tokens = 0

    def sample(self, prompts, greedy=False):
        out = []
        for p in prompts:
            action = 0 if greedy else self.random.randrange(4)
            text = f"<signal>{PHASES[action]}</signal>"
            out.append(Response(p, [], [action + 1], text, action))
        self.calls += len(out)
        self.generated_tokens += len(out)
        return out


class LLMPolicy:
    def __init__(self, config, trainable=False, adapter=None):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from .environment import enable_local_dependencies

        enable_local_dependencies()
        from minimal_cityflow_grpo import load_policy

        if not torch.cuda.is_available():
            raise RuntimeError(
                "LLM commands require CUDA; use --scripted only for infrastructure checks"
            )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 GPU required")
        self.torch, self.config = torch, config
        self.device = torch.device("cuda:0")
        self.model, self.tokenizer = load_policy(
            config["model"],
            device=self.device,
            local_files_only=config.get("local_files_only", False),
        )
        if adapter:
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter,
                is_trainable=trainable,
                local_files_only=config.get("local_files_only", False),
            )
        elif trainable:
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=config["lora_rank"],
                    lora_alpha=2 * config["lora_rank"],
                    lora_dropout=0.0,
                    target_modules=config["lora_target_modules"],
                    task_type="CAUSAL_LM",
                    bias="none",
                ),
            )
        # Identical stochastic model density in generation/old/new logps.
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.Dropout):
                layer.p = 0.0
        if trainable:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.model.enable_input_require_grads()
        self.model.eval()
        self.calls = self.generated_tokens = 0

    def render(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def sample(self, prompts, greedy=False):
        t, tokenizer = self.torch, self.tokenizer
        self.model.eval()
        out = []
        batch_size = self.config["generation_batch_size"]
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            strings = [self.render(p) for p in batch]
            encoded = tokenizer(
                strings, padding=True, return_tensors="pt", add_special_tokens=False
            )
            if encoded.input_ids.shape[1] > self.config["max_prompt_tokens"]:
                raise ValueError(
                    "Prompt too long; increase max_prompt_tokens (traffic state is never truncated)"
                )
            encoded = encoded.to(self.device)
            generation = dict(
                do_sample=not greedy,
                max_new_tokens=self.config["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                min_new_tokens=0,
            )
            if not greedy:
                generation.update(temperature=1.0, top_p=1.0, top_k=0, typical_p=1.0)
            with t.inference_mode():
                ids = self.model.generate(**encoded, **generation)
            eos = self.model.generation_config.eos_token_id
            eos = {eos} if isinstance(eos, int) else set(eos or [])
            for i, messages in enumerate(batch):
                prompt_ids = encoded.input_ids[i][
                    encoded.attention_mask[i].bool()
                ].tolist()
                tokens = ids[i, encoded.input_ids.shape[1] :].tolist()
                stop = next(
                    (j + 1 for j, token in enumerate(tokens) if token in eos),
                    len(tokens),
                )
                tokens = tokens[
                    :stop
                ]  # Include exactly the first EOS; remove generated batch padding.
                if not tokens:
                    raise RuntimeError("Model generated an empty completion")
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                truncated = (
                    len(tokens) == self.config["max_new_tokens"]
                    and tokens[-1] not in eos
                )
                out.append(
                    Response(
                        messages,
                        prompt_ids,
                        tokens,
                        text,
                        parse_action(text),
                        truncated,
                    )
                )
                self.generated_tokens += len(tokens)
        self.calls += len(out)
        return out

    def token_logps(self, response, reference=False):
        t = self.torch
        ids = t.tensor(
            [response.prompt_ids + response.completion_ids], device=self.device
        )
        context = (
            self.model.disable_adapter() if reference else contextlib.nullcontext()
        )
        with context:
            # Keep the one preceding prediction position plus the completion.
            logits = self.model(
                input_ids=ids,
                attention_mask=t.ones_like(ids),
                use_cache=False,
                logits_to_keep=len(response.completion_ids) + 1,
            ).logits
            logits = logits[:, -(len(response.completion_ids) + 1) : -1, :].float()
            labels = t.tensor([response.completion_ids], device=self.device)
            # Cross entropy avoids materializing a second full-vocabulary tensor.
            return -t.nn.functional.cross_entropy(
                logits[0], labels[0], reduction="none"
            )

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
