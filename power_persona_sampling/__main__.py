from __future__ import annotations

import argparse
from pathlib import Path

from .config import PersonaSpec, SamplerConfig
from .batch_sampler import BatchMHSampler
from .datasets import ImRightDatasetBuilder, ImRightGlobalDatasetBuilder, iter_problems_jsonl
from .io_utils import write_jsonl
from .extract import PersonaVectorExtractor, iter_examples_jsonl
from .eval import extract_gold_int, gsm8k_prompt_style, iter_gsm8k
from .eval_runner import EvalItem, run_gsm8k_eval, rows_to_jsonl
from .lm.hf import HFModelConfig, HuggingFaceLM
from .lm.base import SamplingParams
from .lm.steering import SteeringVector
from .persona import PersonaVectorSet
from .sampler import MHSampler
from .prompts import read_prompts_txt


def _parse_persona_spec(s: str) -> PersonaSpec:
    parts = s.split(":")
    if len(parts) not in (2, 4):
        raise argparse.ArgumentTypeError("persona spec must be name:layer or name:layer:beta:lam")
    name = parts[0]
    layer = int(parts[1])
    if len(parts) == 2:
        return PersonaSpec(name=name, layer=layer)
    return PersonaSpec(name=name, layer=layer, beta=float(parts[2]), lam=float(parts[3]))


def _parse_persona_weight(s: str) -> tuple[str, float, float]:
    parts = s.split(":")
    if len(parts) == 1:
        return parts[0], 1.0, 1.0
    if len(parts) == 3:
        return parts[0], float(parts[1]), float(parts[2])
    raise argparse.ArgumentTypeError("persona weight must be name or name:beta:lam")


def _default_device() -> str:
    try:
        import torch

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="power-persona-sampling")
    sub = p.add_subparsers(dest="cmd", required=True)

    info = sub.add_parser("model-info", help="print basic model info (layers, device suggestions)")
    info.add_argument("--model", required=True)
    info.add_argument("--device", default=None)
    info.add_argument("--dtype", default=None)

    extract = sub.add_parser("extract-persona", help="extract persona vectors from JSONL data")
    extract.add_argument("--model", required=True)
    extract.add_argument("--data", required=True, help="JSONL with {prompt,response,persona,label}")
    extract.add_argument("--out", required=True, help="output .pt file")
    extract.add_argument("--persona", action="append", required=True, type=_parse_persona_spec)
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--no-normalize", action="store_true")
    extract.add_argument("--device", default=None)
    extract.add_argument("--dtype", default=None, help="e.g. float16, bfloat16, float32")

    sample = sub.add_parser("sample", help="run MH sampling for a prompt")
    sample.add_argument("--model", required=True)
    sample.add_argument("--persona-vectors", default=None, help="persona vectors .pt file")
    sample.add_argument("--no-persona", action="store_true", help="ignore persona vectors entirely")
    sample.add_argument("--persona", action="append", default=None, help="persona name (repeatable)")
    sample.add_argument("--prompt", default=None)
    sample.add_argument("--prompt-file", default=None)
    sample.add_argument("--steps", type=int, default=200)
    sample.add_argument("--alpha", type=float, default=1.0)
    sample.add_argument("--max-new-tokens", type=int, default=256)
    sample.add_argument("--temperature", type=float, default=1.0)
    sample.add_argument("--top-p", type=float, default=1.0)
    sample.add_argument("--seed", type=int, default=None)
    sample.add_argument("--burn-in", type=int, default=0)
    sample.add_argument("--thin", type=int, default=1)
    sample.add_argument("--device", default=None)
    sample.add_argument("--dtype", default=None)
    sample.add_argument(
        "--persona-spec",
        action="append",
        default=None,
        help="repeatable: name[:beta:lam] (layer inferred from persona vectors file)",
    )

    compare = sub.add_parser("compare-batch", help="baseline vs MH for multiple prompts")
    compare.add_argument("--model", required=True)
    compare.add_argument("--prompts-file", required=True, help="text file; prompts separated by a line with ---")
    compare.add_argument("--out", required=True, help="markdown output file")
    compare.add_argument("--device", default=None)
    compare.add_argument("--dtype", default=None)
    compare.add_argument("--max-new-tokens", type=int, default=512)
    compare.add_argument("--temperature", type=float, default=0.7)
    compare.add_argument("--top-p", type=float, default=0.95)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--mh-steps", type=int, default=10)
    compare.add_argument("--alpha", type=float, default=3.0)
    compare.add_argument("--persona-vectors", default=None, help="optional persona vectors .pt file")
    compare.add_argument(
        "--persona-spec",
        action="append",
        default=None,
        help="repeatable: name[:beta:lam] (layer inferred from persona vectors file)",
    )

    imr = sub.add_parser("make-im-right-jsonl", help="build a correctness-labeled dataset for persona extraction")
    imr.add_argument("--model", required=True)
    imr.add_argument("--problems", required=True, help="JSONL with {prompt,answer[,id]}")
    imr.add_argument("--out", required=True, help="output JSONL with {prompt,response,persona,label,...}")
    imr.add_argument("--persona-name", default="im_right")
    imr.add_argument("--target-pos", type=int, default=4)
    imr.add_argument("--target-neg", type=int, default=4)
    imr.add_argument("--batch-size", type=int, default=8)
    imr.add_argument("--max-rounds", type=int, default=50)
    imr.add_argument("--max-new-tokens", type=int, default=256)
    imr.add_argument("--temperature", type=float, default=0.7)
    imr.add_argument("--top-p", type=float, default=0.95)
    imr.add_argument("--device", default=None)
    imr.add_argument("--dtype", default=None)

    imr2 = sub.add_parser(
        "make-im-right-jsonl-global",
        help="build a correctness-labeled dataset globally (targets total pos/neg, not per-problem)",
    )
    imr2.add_argument("--model", required=True)
    imr2.add_argument("--problems", required=True, help="JSONL with {prompt,answer[,id]}")
    imr2.add_argument("--out", required=True, help="output JSONL with {prompt,response,persona,label,...}")
    imr2.add_argument("--persona-name", default="im_right")
    imr2.add_argument("--target-pos", type=int, default=100)
    imr2.add_argument("--target-neg", type=int, default=100)
    imr2.add_argument("--batch-size", type=int, default=8)
    imr2.add_argument("--max-rounds-per-problem", type=int, default=20)
    imr2.add_argument("--max-new-tokens", type=int, default=256)
    imr2.add_argument("--temperature", type=float, default=0.9)
    imr2.add_argument("--top-p", type=float, default=0.95)
    imr2.add_argument("--device", default=None)
    imr2.add_argument("--dtype", default=None)

    gsm = sub.add_parser("eval-gsm8k", help="evaluate baseline vs MH on GSM8K")
    gsm.add_argument("--model", required=True)
    gsm.add_argument("--split", default="test")
    gsm.add_argument("--limit", type=int, default=100)
    gsm.add_argument("--seed", type=int, default=0)
    gsm.add_argument("--batch-size", type=int, default=8)
    gsm.add_argument("--out", required=True, help="output JSONL path")
    gsm.add_argument("--device", default=None)
    gsm.add_argument("--dtype", default=None)
    gsm.add_argument("--max-new-tokens", type=int, default=256)
    gsm.add_argument("--temperature", type=float, default=0.7)
    gsm.add_argument("--top-p", type=float, default=0.95)
    gsm.add_argument("--mh-steps", type=int, default=30)
    gsm.add_argument("--alpha", type=float, default=3.0)
    gsm.add_argument("--prompt-style", default="short", choices=["short", "cot"])
    gsm.add_argument("--persona-vectors", default=None)
    gsm.add_argument("--persona-spec", action="append", default=None)

    gsm2 = sub.add_parser("eval-gsm8k-dual", help="evaluate baseline vs MH (no persona) vs MH (persona)")
    gsm2.add_argument("--model", required=True)
    gsm2.add_argument("--split", default="test")
    gsm2.add_argument("--limit", type=int, default=30)
    gsm2.add_argument("--seed", type=int, default=0)
    gsm2.add_argument("--batch-size", type=int, default=2)
    gsm2.add_argument("--out", required=True, help="output JSONL path")
    gsm2.add_argument("--device", default=None)
    gsm2.add_argument("--dtype", default=None)
    gsm2.add_argument("--max-new-tokens", type=int, default=256)
    gsm2.add_argument("--temperature", type=float, default=0.7)
    gsm2.add_argument("--top-p", type=float, default=0.95)
    gsm2.add_argument("--mh-steps", type=int, default=10)
    gsm2.add_argument("--alpha", type=float, default=3.0)
    gsm2.add_argument("--prompt-style", default="short", choices=["short", "cot"])
    gsm2.add_argument("--persona-vectors", required=True)
    gsm2.add_argument("--persona-spec", action="append", default=None)
    gsm2.add_argument("--chunk-size", type=int, default=None, help="process eval prompts in chunks to control memory use")

    gsm3 = sub.add_parser("eval-gsm8k-persona", help="evaluate MH with persona potentials only (no baseline outputs)")
    gsm3.add_argument("--model", required=True)
    gsm3.add_argument("--split", default="test")
    gsm3.add_argument("--limit", type=int, default=50)
    gsm3.add_argument("--seed", type=int, default=0)
    gsm3.add_argument("--batch-size", type=int, default=2)
    gsm3.add_argument("--chunk-size", type=int, default=8)
    gsm3.add_argument("--out", required=True, help="output JSONL path")
    gsm3.add_argument("--device", default=None)
    gsm3.add_argument("--dtype", default=None)
    gsm3.add_argument("--max-new-tokens", type=int, default=256)
    gsm3.add_argument("--temperature", type=float, default=0.7)
    gsm3.add_argument("--top-p", type=float, default=0.95)
    gsm3.add_argument("--mh-steps", type=int, default=10)
    gsm3.add_argument("--alpha", type=float, default=3.0)
    gsm3.add_argument("--prompt-style", default="short", choices=["short", "cot"])
    gsm3.add_argument("--persona-vectors", required=True)
    gsm3.add_argument("--persona-spec", action="append", default=None, required=True)

    steer = sub.add_parser("eval-gsm8k-steer", help="evaluate baseline vs activation steering (+/-) on GSM8K")
    steer.add_argument("--model", required=True)
    steer.add_argument("--split", default="test")
    steer.add_argument("--search-limit", type=int, default=200, help="number of GSM8K problems to scan for baseline-correct")
    steer.add_argument("--target-correct", type=int, default=50, help="number of baseline-correct problems to report")
    steer.add_argument("--seed", type=int, default=0)
    steer.add_argument("--out", required=True, help="output JSONL path (baseline-correct subset only)")
    steer.add_argument("--device", default=None)
    steer.add_argument("--dtype", default=None)
    steer.add_argument("--max-new-tokens", type=int, default=256)
    steer.add_argument("--temperature", type=float, default=0.0)
    steer.add_argument("--top-p", type=float, default=1.0)
    steer.add_argument("--prompt-style", default="short", choices=["short", "cot"])
    steer.add_argument("--persona-vectors", required=True)
    steer.add_argument("--persona-name", default="im_right")
    steer.add_argument("--scale", type=float, default=5.0, help="steering magnitude; applies +/-scale")
    steer.add_argument("--layer", type=int, default=None, help="override layer (default: from persona vectors file)")
    steer.add_argument("--positions", default="last", choices=["last", "all"], help="apply steering to last token or all tokens in the layer output")
    steer.add_argument("--chunk-size", type=int, default=8, help="prompt chunk size for generation")
    steer.add_argument("--greedy", action="store_true", help="use greedy decoding (ignores temperature/top-p)")

    steer2 = sub.add_parser("steer-flip", help="measure how often negative steering flips baseline-correct answers")
    steer2.add_argument("--model", required=True)
    steer2.add_argument("--problems", required=True, help="JSONL with {prompt,answer[,id]}")
    steer2.add_argument("--out", required=True, help="output JSONL path")
    steer2.add_argument("--seed", type=int, default=0)
    steer2.add_argument("--device", default=None)
    steer2.add_argument("--dtype", default=None)
    steer2.add_argument("--max-new-tokens", type=int, default=256)
    steer2.add_argument("--temperature", type=float, default=0.0)
    steer2.add_argument("--top-p", type=float, default=1.0)
    steer2.add_argument("--greedy", action="store_true")
    steer2.add_argument("--chunk-size", type=int, default=8)
    steer2.add_argument("--max-problems", type=int, default=None, help="optional cap on number of problems processed")
    steer2.add_argument("--target-correct", type=int, default=50, help="number of baseline-correct problems to evaluate")
    steer2.add_argument("--persona-vectors", required=True)
    steer2.add_argument("--persona-name", default="im_right")
    steer2.add_argument("--scale", type=float, default=5.0)
    steer2.add_argument("--layer", type=int, default=None, help="override layer (default: from persona vectors file)")
    steer2.add_argument("--positions", default="last", choices=["last", "all"])

    steer3 = sub.add_parser("compare-steer", help="write baseline vs (+/-) steering generations for prompts")
    steer3.add_argument("--model", required=True)
    steer3.add_argument("--prompts-file", required=True, help="text file; prompts separated by a line with ---")
    steer3.add_argument("--out", required=True, help="markdown output file")
    steer3.add_argument("--seed", type=int, default=0)
    steer3.add_argument("--device", default=None)
    steer3.add_argument("--dtype", default=None)
    steer3.add_argument("--max-new-tokens", type=int, default=256)
    steer3.add_argument("--temperature", type=float, default=0.7)
    steer3.add_argument("--top-p", type=float, default=0.95)
    steer3.add_argument("--greedy", action="store_true")
    steer3.add_argument("--chunk-size", type=int, default=8)
    steer3.add_argument("--persona-vectors", required=True)
    steer3.add_argument("--persona-name", default="im_right")
    steer3.add_argument("--scale", type=float, default=5.0)
    steer3.add_argument("--layer", type=int, default=None)
    steer3.add_argument("--positions", default="last", choices=["last", "all"])

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "model-info":
        lm = HuggingFaceLM(
            HFModelConfig(
                model_name_or_path=args.model,
                device=args.device or _default_device(),
                torch_dtype=args.dtype,
            )
        )
        n = getattr(lm._model.config, "num_hidden_layers", None)  # noqa: SLF001
        hs = "unknown"
        if n is not None:
            hs = f"{int(n) + 1} (including embeddings as layer 0)"
        print(f"device={lm.device}")
        print(f"num_hidden_layers={n}")
        print(f"hidden_states_indices={hs}")
        return

    if args.cmd == "extract-persona":
        device = args.device or _default_device()
        lm = HuggingFaceLM(
            HFModelConfig(
                model_name_or_path=args.model,
                device=device,
                torch_dtype=args.dtype,
            )
        )
        specs = list(args.persona)
        extractor = PersonaVectorExtractor(lm=lm, specs=specs, batch_size=args.batch_size)
        vecs = extractor.extract(iter_examples_jsonl(args.data), normalize=not args.no_normalize)
        vecs.save(args.out)
        return

    if args.cmd == "sample":
        if (args.prompt is None) == (args.prompt_file is None):
            raise SystemExit("provide exactly one of --prompt or --prompt-file")
        prompt = args.prompt
        if prompt is None:
            prompt = Path(args.prompt_file).read_text(encoding="utf-8")

        lm = HuggingFaceLM(
            HFModelConfig(
                model_name_or_path=args.model,
                device=args.device or _default_device(),
                torch_dtype=args.dtype,
            )
        )
        if args.no_persona:
            vectors = PersonaVectorSet([])
            specs: list[PersonaSpec] = []
        else:
            if args.persona_vectors is None:
                raise SystemExit("provide --persona-vectors or set --no-persona")
            vectors = PersonaVectorSet.load(args.persona_vectors)
            if args.persona_spec:
                parsed = [_parse_persona_weight(s) for s in args.persona_spec]
                specs = [
                    PersonaSpec(name=n, layer=vectors.get(n).layer, beta=beta, lam=lam)
                    for n, beta, lam in parsed
                ]
            else:
                names = args.persona or vectors.names()
                specs = [PersonaSpec(name=n, layer=vectors.get(n).layer) for n in names]
        cfg = SamplerConfig(
            alpha=float(args.alpha),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            seed=args.seed,
            burn_in=int(args.burn_in),
            thin=int(args.thin),
        )
        sampler = MHSampler(lm=lm, persona_vectors=vectors, persona_specs=specs, cfg=cfg)
        res = sampler.sample(prompt, num_steps=int(args.steps), show_progress=True)

        accepted = sum(1 for s in res.stats if s.accepted)
        rate = accepted / max(1, len(res.stats))
        text = sampler.decode(res.final)

        print(f"accept_rate={rate:.3f}")
        for k, v in res.final.persona_scores.items():
            print(f"{k}={v:.4f}")
        print("---")
        print(text)
        return

    if args.cmd == "compare-batch":
        prompts = read_prompts_txt(args.prompts_file)
        if not prompts:
            raise SystemExit("no prompts found")

        lm = HuggingFaceLM(
            HFModelConfig(
                model_name_or_path=args.model,
                device=args.device or _default_device(),
                torch_dtype=args.dtype,
            )
        )

        params = SamplerConfig(
            alpha=1.0,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            seed=int(args.seed),
        )
        base = BatchMHSampler(
            lm=lm,
            persona_vectors=PersonaVectorSet([]),
            persona_specs=[],
            cfg=params,
        )
        base_responses = [base.decode(s) for s in base.sample(prompts, num_steps=0, show_progress=False).finals]

        persona_vectors = PersonaVectorSet([])
        persona_specs: list[PersonaSpec] = []
        if args.persona_vectors is not None:
            persona_vectors = PersonaVectorSet.load(args.persona_vectors)
            if args.persona_spec:
                parsed = [_parse_persona_weight(s) for s in args.persona_spec]
                persona_specs = [
                    PersonaSpec(name=n, layer=persona_vectors.get(n).layer, beta=beta, lam=lam)
                    for n, beta, lam in parsed
                ]
            else:
                persona_specs = [
                    PersonaSpec(name=n, layer=persona_vectors.get(n).layer) for n in persona_vectors.names()
                ]

        mh_cfg = SamplerConfig(
            alpha=float(args.alpha),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            seed=int(args.seed),
        )
        mh = BatchMHSampler(
            lm=lm,
            persona_vectors=persona_vectors,
            persona_specs=persona_specs,
            cfg=mh_cfg,
        )
        mh_out = mh.sample(prompts, num_steps=int(args.mh_steps), show_progress=True)
        mh_responses = [mh.decode(s) for s in mh_out.finals]

        accept_rates = []
        for i in range(len(prompts)):
            accepted = sum(1 for st in mh_out.stats if st.accepted[i])
            accept_rates.append(accepted / max(1, len(mh_out.stats)))

        lines: list[str] = []
        fence = "````"
        lines.append(f"# Batch Compare ({args.model})")
        lines.append("")
        lines.append(f"- device: `{args.device or _default_device()}`")
        lines.append(f"- max_new_tokens: `{args.max_new_tokens}`")
        lines.append(f"- temperature: `{args.temperature}`")
        lines.append(f"- top_p: `{args.top_p}`")
        lines.append(f"- MH: `steps={args.mh_steps}`, `alpha={args.alpha}`")
        lines.append("")

        for j, prompt in enumerate(prompts, start=1):
            lines.append(f"## Prompt {j}")
            lines.append("")
            lines.append(fence)
            lines.append(prompt.strip())
            lines.append(fence)
            lines.append("")
            lines.append("### Base (p0)")
            lines.append("")
            lines.append(fence)
            base_txt = base_responses[j - 1].strip()
            lines.append(base_txt if base_txt else "<EMPTY>")
            lines.append(fence)
            lines.append("")
            extra = ""
            if mh_out.finals[j - 1].persona_scores:
                extra = " " + " ".join(
                    f"{k}={v:.3f}" for k, v in mh_out.finals[j - 1].persona_scores.items()
                )
            lines.append(
                f"### MH (alpha={args.alpha}, steps={args.mh_steps}, accept_rate={accept_rates[j-1]:.3f}){extra}"
            )
            lines.append("")
            lines.append(fence)
            mh_txt = mh_responses[j - 1].strip()
            lines.append(mh_txt if mh_txt else "<EMPTY>")
            lines.append(fence)
            lines.append("")

        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
        return

    if args.cmd == "make-im-right-jsonl":
        lm = HuggingFaceLM(
            HFModelConfig(
                model_name_or_path=args.model,
                device=args.device or _default_device(),
                torch_dtype=args.dtype,
            )
        )
        problems = list(iter_problems_jsonl(args.problems))
        builder = ImRightDatasetBuilder(
            lm=lm,
            persona_name=str(args.persona_name),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
            batch_size=int(args.batch_size),
            max_rounds=int(args.max_rounds),
        )
        rows = builder.build(
            problems,
            target_pos=int(args.target_pos),
            target_neg=int(args.target_neg),
            show_progress=True,
        )
        write_jsonl(args.out, rows)
        print(f"wrote {args.out} ({len(rows)} rows)")
        return

    if args.cmd == "make-im-right-jsonl-global":
        lm = HuggingFaceLM(
            HFModelConfig(
                model_name_or_path=args.model,
                device=args.device or _default_device(),
                torch_dtype=args.dtype,
            )
        )
        problems = list(iter_problems_jsonl(args.problems))
        builder = ImRightGlobalDatasetBuilder(
            lm=lm,
            persona_name=str(args.persona_name),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
            batch_size=int(args.batch_size),
            max_rounds_per_problem=int(args.max_rounds_per_problem),
        )
        rows = builder.build(
            problems,
            target_pos=int(args.target_pos),
            target_neg=int(args.target_neg),
            show_progress=True,
        )
        write_jsonl(args.out, rows)
        print(f"wrote {args.out} ({len(rows)} rows)")
        return

    if args.cmd == "eval-gsm8k":
        lm = HuggingFaceLM(
            HFModelConfig(
                model_name_or_path=args.model,
                device=args.device or _default_device(),
                torch_dtype=args.dtype,
            )
        )
        examples = list(iter_gsm8k(split=str(args.split)))
        rng = __import__("random").Random(int(args.seed))
        rng.shuffle(examples)
        examples = examples[: int(args.limit)]

        items = [
            EvalItem(
                prompt=gsm8k_prompt_style(ex.question, style=str(args.prompt_style)),
                gold=extract_gold_int(ex.answer),
            )
            for ex in examples
        ]

        pv = PersonaVectorSet([])
        ps: list[PersonaSpec] = []
        if args.persona_vectors is not None:
            pv = PersonaVectorSet.load(args.persona_vectors)
            if args.persona_spec:
                parsed = [_parse_persona_weight(s) for s in args.persona_spec]
                ps = [PersonaSpec(name=n, layer=pv.get(n).layer, beta=beta, lam=lam) for n, beta, lam in parsed]
            else:
                ps = [PersonaSpec(name=n, layer=pv.get(n).layer) for n in pv.names()]

        base_params = SamplingParams(
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
        )
        mh_cfg = SamplerConfig(
            alpha=float(args.alpha),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            seed=int(args.seed),
        )
        summary, rows = run_gsm8k_eval(
            lm=lm,
            items=items,
            base_params=base_params,
            mh_cfg=mh_cfg,
            mh_steps=int(args.mh_steps),
            persona_vectors=pv,
            persona_specs=ps,
            batch_size=int(args.batch_size),
        )
        write_jsonl(args.out, rows_to_jsonl(rows))
        print(f"n={summary.n} base_acc={summary.base_acc:.3f} mh_acc={summary.mh_acc:.3f}")
        print(f"wrote {args.out}")
        return

    if args.cmd == "eval-gsm8k-dual":
        lm = HuggingFaceLM(
            HFModelConfig(
                model_name_or_path=args.model,
                device=args.device or _default_device(),
                torch_dtype=args.dtype,
            )
        )
        examples = list(iter_gsm8k(split=str(args.split)))
        rng = __import__("random").Random(int(args.seed))
        rng.shuffle(examples)
        examples = examples[: int(args.limit)]

        items = [
            EvalItem(
                prompt=gsm8k_prompt_style(ex.question, style=str(args.prompt_style)),
                gold=extract_gold_int(ex.answer),
            )
            for ex in examples
        ]

        base_params = SamplingParams(
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
        )
        mh_cfg = SamplerConfig(
            alpha=float(args.alpha),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            seed=int(args.seed),
        )

        base_sampler = BatchMHSampler(
            lm=lm,
            persona_vectors=PersonaVectorSet([]),
            persona_specs=[],
            cfg=mh_cfg,
        )

        pv = PersonaVectorSet.load(args.persona_vectors)
        if args.persona_spec:
            parsed = [_parse_persona_weight(s) for s in args.persona_spec]
            ps = [PersonaSpec(name=n, layer=pv.get(n).layer, beta=beta, lam=lam) for n, beta, lam in parsed]
        else:
            ps = [PersonaSpec(name=n, layer=pv.get(n).layer) for n in pv.names()]
        persona_sampler = BatchMHSampler(lm=lm, persona_vectors=pv, persona_specs=ps, cfg=mh_cfg)

        def _accept_rates(batch_stats: list[BatchStepStats]) -> list[float]:
            if not batch_stats:
                return []
            n = len(batch_stats[0].accepted)
            counts = [0.0] * n
            for st in batch_stats:
                for i, acc in enumerate(st.accepted):
                    counts[i] += 1.0 if acc else 0.0
            denom = len(batch_stats) if batch_stats else 1
            return [c / denom for c in counts]

        chunk_size = int(args.chunk_size) if args.chunk_size else len(items)
        if chunk_size <= 0:
            raise SystemExit("--chunk-size must be positive")

        from .correctness import extract_gsm8k_answer, is_correct_gsm8k

        rows = []
        for start in range(0, len(items), chunk_size):
            batch = items[start : start + chunk_size]
            prompts = [it.prompt for it in batch]
            prompt_ids = [lm.encode(p) for p in prompts]
            suffixes = lm.sample_suffix_batch(prompt_ids, params=base_params)
            base_texts = [
                lm.decode(lm.ensure_ended(s, max_new_tokens=base_params.max_new_tokens)) for s in suffixes
            ]

            base_mh = base_sampler.sample(
                prompts,
                num_steps=int(args.mh_steps),
                init_responses=base_texts,
                show_progress=True,
            )
            base_mh_texts = [base_sampler.decode(s) for s in base_mh.finals]

            persona_mh = persona_sampler.sample(
                prompts,
                num_steps=int(args.mh_steps),
                init_responses=base_texts,
                show_progress=True,
            )
            persona_mh_texts = [persona_sampler.decode(s) for s in persona_mh.finals]

            base_accept = _accept_rates(base_mh.stats)
            persona_accept = _accept_rates(persona_mh.stats)

            for i, it in enumerate(batch):
                acc0 = float(base_accept[i]) if i < len(base_accept) else 0.0
                acc1 = float(persona_accept[i]) if i < len(persona_accept) else 0.0
                rows.append(
                    {
                        "prompt": it.prompt,
                        "gold": it.gold,
                        "base_text": base_texts[i],
                        "base_pred": extract_gsm8k_answer(base_texts[i]),
                        "base_correct": is_correct_gsm8k(base_texts[i], it.gold),
                        "mh_text": base_mh_texts[i],
                        "mh_pred": extract_gsm8k_answer(base_mh_texts[i]),
                        "mh_correct": is_correct_gsm8k(base_mh_texts[i], it.gold),
                        "mh_accept_rate": acc0,
                        "mh_persona_scores": dict(base_mh.finals[i].persona_scores),
                        "mh_imright_text": persona_mh_texts[i],
                        "mh_imright_pred": extract_gsm8k_answer(persona_mh_texts[i]),
                        "mh_imright_correct": is_correct_gsm8k(persona_mh_texts[i], it.gold),
                        "mh_imright_accept_rate": acc1,
                        "mh_imright_persona_scores": dict(persona_mh.finals[i].persona_scores),
                    }
                )

        base_acc = sum(1 for r in rows if r["base_correct"]) / max(1, len(rows))
        mh_acc = sum(1 for r in rows if r["mh_correct"]) / max(1, len(rows))
        mh_im_acc = sum(1 for r in rows if r["mh_imright_correct"]) / max(1, len(rows))
        write_jsonl(args.out, rows)
        print(f"n={len(rows)} base_acc={base_acc:.3f} mh_acc={mh_acc:.3f} mh_imright_acc={mh_im_acc:.3f}")
        print(f"wrote {args.out}")
        return

    if args.cmd == "eval-gsm8k-persona":
        lm = HuggingFaceLM(
            HFModelConfig(
                model_name_or_path=args.model,
                device=args.device or _default_device(),
                torch_dtype=args.dtype,
            )
        )
        examples = list(iter_gsm8k(split=str(args.split)))
        rng = __import__("random").Random(int(args.seed))
        rng.shuffle(examples)
        examples = examples[: int(args.limit)]

        items = [
            EvalItem(
                prompt=gsm8k_prompt_style(ex.question, style=str(args.prompt_style)),
                gold=extract_gold_int(ex.answer),
            )
            for ex in examples
        ]

        base_params = SamplingParams(
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
        )
        mh_cfg = SamplerConfig(
            alpha=float(args.alpha),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            seed=int(args.seed),
        )

        pv = PersonaVectorSet.load(args.persona_vectors)
        parsed = [_parse_persona_weight(s) for s in args.persona_spec]
        ps = [PersonaSpec(name=n, layer=pv.get(n).layer, beta=beta, lam=lam) for n, beta, lam in parsed]
        persona_sampler = BatchMHSampler(lm=lm, persona_vectors=pv, persona_specs=ps, cfg=mh_cfg)

        from .correctness import extract_gsm8k_answer, is_correct_gsm8k

        chunk_size = int(args.chunk_size)
        if chunk_size <= 0:
            raise SystemExit("--chunk-size must be positive")

        def _accept_rates(batch_stats: list[BatchStepStats]) -> list[float]:
            if not batch_stats:
                return []
            n = len(batch_stats[0].accepted)
            counts = [0.0] * n
            for st in batch_stats:
                for i, acc in enumerate(st.accepted):
                    counts[i] += 1.0 if acc else 0.0
            denom = len(batch_stats) if batch_stats else 1
            return [c / denom for c in counts]

        rows = []
        for start in range(0, len(items), chunk_size):
            batch = items[start : start + chunk_size]
            prompts = [it.prompt for it in batch]
            prompt_ids = [lm.encode(p) for p in prompts]
            init_suffixes = lm.sample_suffix_batch(prompt_ids, params=base_params)
            init_texts = [
                lm.decode(lm.ensure_ended(s, max_new_tokens=base_params.max_new_tokens)) for s in init_suffixes
            ]

            persona_mh = persona_sampler.sample(
                prompts,
                num_steps=int(args.mh_steps),
                init_responses=init_texts,
                show_progress=True,
            )
            persona_mh_texts = [persona_sampler.decode(s) for s in persona_mh.finals]
            persona_accept = _accept_rates(persona_mh.stats)

            for i, it in enumerate(batch):
                acc = float(persona_accept[i]) if i < len(persona_accept) else 0.0
                rows.append(
                    {
                        "prompt": it.prompt,
                        "gold": it.gold,
                        "init_text": init_texts[i],
                        "text": persona_mh_texts[i],
                        "pred": extract_gsm8k_answer(persona_mh_texts[i]),
                        "correct": is_correct_gsm8k(persona_mh_texts[i], it.gold),
                        "accept_rate": acc,
                        "persona_scores": dict(persona_mh.finals[i].persona_scores),
                    }
                )

        acc = sum(1 for r in rows if r["correct"]) / max(1, len(rows))
        write_jsonl(args.out, rows)
        print(f"n={len(rows)} acc={acc:.3f}")
        print(f"wrote {args.out}")
        return

    if args.cmd == "eval-gsm8k-steer":
        lm = HuggingFaceLM(
            HFModelConfig(
                model_name_or_path=args.model,
                device=args.device or _default_device(),
                torch_dtype=args.dtype,
            )
        )
        examples = list(iter_gsm8k(split=str(args.split)))
        rng = __import__("random").Random(int(args.seed))
        rng.shuffle(examples)
        examples = examples[: int(args.search_limit)]

        items = [
            EvalItem(
                prompt=gsm8k_prompt_style(ex.question, style=str(args.prompt_style)),
                gold=extract_gold_int(ex.answer),
            )
            for ex in examples
        ]
        prompts = [it.prompt for it in items]

        params = SamplingParams(
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
        )
        chunk_size = int(args.chunk_size)
        if chunk_size <= 0:
            raise SystemExit("--chunk-size must be positive")

        from .correctness import extract_gsm8k_answer, is_correct_gsm8k

        base_texts: list[str] = []
        for start in range(0, len(prompts), chunk_size):
            batch_prompts = prompts[start : start + chunk_size]
            batch_ids = [lm.encode(p) for p in batch_prompts]
            suffixes = lm.sample_suffix_batch_custom(
                batch_ids,
                params=params,
                steering=(),
                seed=int(args.seed) + start,
                greedy=bool(args.greedy),
            )
            base_texts.extend(
                [lm.decode(lm.ensure_ended(s, max_new_tokens=params.max_new_tokens)) for s in suffixes]
            )

        base_correct = [is_correct_gsm8k(t, items[i].gold) for i, t in enumerate(base_texts)]
        correct_idx = [i for i, ok in enumerate(base_correct) if ok]
        target = min(int(args.target_correct), len(correct_idx))
        if target <= 0:
            raise SystemExit("no baseline-correct examples found; increase --search-limit or change decoding")
        correct_idx = correct_idx[:target]

        pv = PersonaVectorSet.load(args.persona_vectors)
        vec = pv.get(str(args.persona_name))
        layer = int(args.layer) if args.layer is not None else int(vec.layer)
        scale = float(args.scale)
        v = vec.vector.to(device=lm.device)

        pos = [SteeringVector(layer=layer, vector=v, scale=+scale, positions=str(args.positions))]
        neg = [SteeringVector(layer=layer, vector=v, scale=-scale, positions=str(args.positions))]

        sel_prompts = [prompts[i] for i in correct_idx]
        pos_texts: list[str] = []
        neg_texts: list[str] = []
        for start in range(0, len(sel_prompts), chunk_size):
            batch_prompts = sel_prompts[start : start + chunk_size]
            batch_ids = [lm.encode(p) for p in batch_prompts]
            pos_suf = lm.sample_suffix_batch_custom(
                batch_ids,
                params=params,
                steering=pos,
                seed=int(args.seed) + 100_000 + start,
                greedy=bool(args.greedy),
            )
            neg_suf = lm.sample_suffix_batch_custom(
                batch_ids,
                params=params,
                steering=neg,
                seed=int(args.seed) + 200_000 + start,
                greedy=bool(args.greedy),
            )
            pos_texts.extend([lm.decode(lm.ensure_ended(s, max_new_tokens=params.max_new_tokens)) for s in pos_suf])
            neg_texts.extend([lm.decode(lm.ensure_ended(s, max_new_tokens=params.max_new_tokens)) for s in neg_suf])

        rows = []
        for j, i in enumerate(correct_idx):
            gold = items[i].gold
            rows.append(
                {
                    "source_index": int(i),
                    "prompt": prompts[i],
                    "gold": gold,
                    "baseline_text": base_texts[i],
                    "baseline_pred": extract_gsm8k_answer(base_texts[i]),
                    "baseline_correct": True,
                    "steer_pos_scale": scale,
                    "steer_pos_text": pos_texts[j],
                    "steer_pos_pred": extract_gsm8k_answer(pos_texts[j]),
                    "steer_pos_correct": is_correct_gsm8k(pos_texts[j], gold),
                    "steer_neg_scale": -scale,
                    "steer_neg_text": neg_texts[j],
                    "steer_neg_pred": extract_gsm8k_answer(neg_texts[j]),
                    "steer_neg_correct": is_correct_gsm8k(neg_texts[j], gold),
                    "persona_name": str(args.persona_name),
                    "layer": layer,
                }
            )

        pos_acc = sum(1 for r in rows if r["steer_pos_correct"]) / max(1, len(rows))
        neg_acc = sum(1 for r in rows if r["steer_neg_correct"]) / max(1, len(rows))
        neg_flip = sum(1 for r in rows if not r["steer_neg_correct"]) / max(1, len(rows))
        write_jsonl(args.out, rows)
        print(f"baseline_correct_subset={len(rows)} pos_acc={pos_acc:.3f} neg_acc={neg_acc:.3f} neg_flip_rate={neg_flip:.3f}")
        print(f"wrote {args.out}")
        return

    if args.cmd == "steer-flip":
        from .correctness import extract_final_answer, is_correct

        lm = HuggingFaceLM(
            HFModelConfig(
                model_name_or_path=args.model,
                device=args.device or _default_device(),
                torch_dtype=args.dtype,
            )
        )
        problems = list(iter_problems_jsonl(str(args.problems)))
        if args.max_problems is not None:
            problems = problems[: int(args.max_problems)]
        if not problems:
            raise SystemExit("no problems found")

        chunk_size = int(args.chunk_size)
        if chunk_size <= 0:
            raise SystemExit("--chunk-size must be positive")

        params = SamplingParams(
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
        )

        pv = PersonaVectorSet.load(args.persona_vectors)
        vec = pv.get(str(args.persona_name))
        layer = int(args.layer) if args.layer is not None else int(vec.layer)
        scale = float(args.scale)
        v = vec.vector.to(device=lm.device)
        pos = [SteeringVector(layer=layer, vector=v, scale=+scale, positions=str(args.positions))]
        neg = [SteeringVector(layer=layer, vector=v, scale=-scale, positions=str(args.positions))]

        prompts = [p.prompt for p in problems]
        answers = [p.answer for p in problems]
        ids = [p.id for p in problems]

        base_texts: list[str] = []
        for start in range(0, len(prompts), chunk_size):
            batch_prompts = prompts[start : start + chunk_size]
            batch_ids = [lm.encode(p) for p in batch_prompts]
            suffixes = lm.sample_suffix_batch_custom(
                batch_ids,
                params=params,
                steering=(),
                seed=int(args.seed) + start,
                greedy=bool(args.greedy),
            )
            base_texts.extend([lm.decode(lm.ensure_ended(s, max_new_tokens=params.max_new_tokens)) for s in suffixes])

        base_ok = [is_correct(base_texts[i], answers[i]) for i in range(len(problems))]
        correct_idx = [i for i, ok in enumerate(base_ok) if ok]
        if not correct_idx:
            raise SystemExit("no baseline-correct problems; try sampling (non-greedy) or adjust max_new_tokens")
        target = min(int(args.target_correct), len(correct_idx))
        correct_idx = correct_idx[:target]

        sel_prompts = [prompts[i] for i in correct_idx]
        pos_texts: list[str] = []
        neg_texts: list[str] = []
        for start in range(0, len(sel_prompts), chunk_size):
            batch_prompts = sel_prompts[start : start + chunk_size]
            batch_ids = [lm.encode(p) for p in batch_prompts]
            pos_suf = lm.sample_suffix_batch_custom(
                batch_ids,
                params=params,
                steering=pos,
                seed=int(args.seed) + 100_000 + start,
                greedy=bool(args.greedy),
            )
            neg_suf = lm.sample_suffix_batch_custom(
                batch_ids,
                params=params,
                steering=neg,
                seed=int(args.seed) + 200_000 + start,
                greedy=bool(args.greedy),
            )
            pos_texts.extend([lm.decode(lm.ensure_ended(s, max_new_tokens=params.max_new_tokens)) for s in pos_suf])
            neg_texts.extend([lm.decode(lm.ensure_ended(s, max_new_tokens=params.max_new_tokens)) for s in neg_suf])

        rows = []
        for j, i in enumerate(correct_idx):
            gold = answers[i]
            rows.append(
                {
                    "id": ids[i],
                    "prompt": prompts[i],
                    "gold": gold,
                    "baseline_text": base_texts[i],
                    "baseline_pred": extract_final_answer(base_texts[i]),
                    "baseline_correct": True,
                    "steer_pos_scale": scale,
                    "steer_pos_text": pos_texts[j],
                    "steer_pos_pred": extract_final_answer(pos_texts[j]),
                    "steer_pos_correct": is_correct(pos_texts[j], gold),
                    "steer_neg_scale": -scale,
                    "steer_neg_text": neg_texts[j],
                    "steer_neg_pred": extract_final_answer(neg_texts[j]),
                    "steer_neg_correct": is_correct(neg_texts[j], gold),
                    "persona_name": str(args.persona_name),
                    "layer": layer,
                    "positions": str(args.positions),
                }
            )

        pos_acc = sum(1 for r in rows if r["steer_pos_correct"]) / max(1, len(rows))
        neg_acc = sum(1 for r in rows if r["steer_neg_correct"]) / max(1, len(rows))
        flip = sum(1 for r in rows if not r["steer_neg_correct"]) / max(1, len(rows))
        write_jsonl(args.out, rows)
        print(f"n={len(rows)} pos_acc={pos_acc:.3f} neg_acc={neg_acc:.3f} neg_flip_rate={flip:.3f}")
        print(f"wrote {args.out}")
        return

    if args.cmd == "compare-steer":
        lm = HuggingFaceLM(
            HFModelConfig(
                model_name_or_path=args.model,
                device=args.device or _default_device(),
                torch_dtype=args.dtype,
            )
        )
        prompts = read_prompts_txt(str(args.prompts_file))
        if not prompts:
            raise SystemExit("no prompts found")

        chunk_size = int(args.chunk_size)
        if chunk_size <= 0:
            raise SystemExit("--chunk-size must be positive")

        params = SamplingParams(
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
        )

        pv = PersonaVectorSet.load(args.persona_vectors)
        vec = pv.get(str(args.persona_name))
        layer = int(args.layer) if args.layer is not None else int(vec.layer)
        scale = float(args.scale)
        v = vec.vector.to(device=lm.device)
        pos = [SteeringVector(layer=layer, vector=v, scale=+scale, positions=str(args.positions))]
        neg = [SteeringVector(layer=layer, vector=v, scale=-scale, positions=str(args.positions))]

        def gen(all_prompts: list[str], steering: list[SteeringVector], seed_off: int) -> list[str]:
            texts: list[str] = []
            for start in range(0, len(all_prompts), chunk_size):
                batch_prompts = all_prompts[start : start + chunk_size]
                batch_ids = [lm.encode(p) for p in batch_prompts]
                suf = lm.sample_suffix_batch_custom(
                    batch_ids,
                    params=params,
                    steering=steering,
                    seed=int(args.seed) + seed_off + start,
                    greedy=bool(args.greedy),
                )
                texts.extend([lm.decode(lm.ensure_ended(s, max_new_tokens=params.max_new_tokens)) for s in suf])
            return texts

        base_texts = gen(prompts, [], 0)
        pos_texts = gen(prompts, pos, 100_000)
        neg_texts = gen(prompts, neg, 200_000)

        out = Path(str(args.out))
        parts: list[str] = []
        parts.append(
            f"# Steering compare\n\npersona={args.persona_name} layer={layer} positions={args.positions} scale={scale}\n"
        )
        for i, p in enumerate(prompts, 1):
            parts.append(f"\n## Prompt {i}\n\n```text\n{p}\n```\n")
            parts.append("\n### Baseline\n\n```text\n" + base_texts[i - 1] + "\n```\n")
            parts.append("\n### +steer\n\n```text\n" + pos_texts[i - 1] + "\n```\n")
            parts.append("\n### -steer\n\n```text\n" + neg_texts[i - 1] + "\n```\n")
        out.write_text("".join(parts), encoding="utf-8")
        print(f"wrote {out}")
        return

    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
