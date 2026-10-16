"""C-only CPU preflight: official codec, real frozen weights, mechanisms, regression."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def original_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("OMP_NUM_THREADS") != "2" \
            or os.environ.get("MKL_NUM_THREADS") != "2":
        raise RuntimeError("Run on C with CUDA_VISIBLE_DEVICES='' and OMP/MKL both 2")
    import numpy as np
    import sklearn
    import torch
    import transformers
    import transformers.models.llama.modeling_llama as llama_source
    import transformers.modeling_attn_mask_utils as mask_source
    import transformers.generation.utils as generation_source
    import transformers.models.llama.tokenization_llama as tokenizer_source
    from transformers import LlamaForCausalLM, LlamaTokenizer
    from gofa_f.chattime import ChatTimeGOFA, HistoryCodec
    from gofa_f.chattime_checks import check_model
    from gofa_f.chattime_stage import code_files, config_file, digest, prepare
    from gofa_f.chattime_task import counts, data_for, seed
    from gofa_f.graphs import build_graph
    from gofa_f.prefix_cache import atomic_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage16_chattime.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, output = config_file(args.config), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    assert not torch.cuda.is_initialized()
    packages = {"torch": torch.__version__, "transformers": transformers.__version__,
                "numpy": np.__version__, "scikit-learn": sklearn.__version__}
    if packages != {"torch": "2.6.0+cu124", "transformers": "4.46.3", "numpy": "1.26.4", "scikit-learn": "1.5.2"}:
        raise ValueError("Unregistered runtime: "+str(packages))
    prepare(config, output)
    command = [sys.executable, "-m", "pytest", "-q", "--junitxml="+str(output/"cpu-tests.xml")]
    result = subprocess.run(command, text=True, capture_output=True)
    (output/"cpu-tests.log").write_text(result.stdout+result.stderr, encoding="utf-8")
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError("CPU regression failed; see cpu-tests.log")
    cases = list(ET.parse(output/"cpu-tests.xml").getroot().iter("testcase"))
    chat_cases = [c for c in cases if any(part.startswith("test_chattime")
                  for part in c.attrib.get("classname", "").split("."))]
    if len(chat_cases) < 25 or any(c.find("skipped") is not None for c in chat_cases):
        raise RuntimeError("New Stage16 tests are missing or skipped")
    upstream = Path(config["weights"]["source_directory"])
    tools = original_module("stage16_official_tools", upstream/"utils/tools.py")
    prompt = original_module("stage16_official_prompt", upstream/"utils/prompt.py")
    tokenizer = LlamaTokenizer.from_pretrained(config["weights"]["directory"], local_files_only=True)
    codec = HistoryCodec(tokenizer)
    examples = [np.arange(12.)/7, np.full(12, 3.5),
                np.array([np.nan, 1., 3., np.nan, 2., 0., 5., 1., 0., 2., 3., 1.])]
    for example in examples:
        scaler = tools.Discretizer()
        official = prompt.getPrompt(flag="prediction", input=tools.Serializer().serialize(scaler.discretize(example)))
        ours, _, _, _, inverse_scale, inverse_offset = codec.serialize(example, np.isfinite(example), 0.)
        if ours != official:
            raise AssertionError("Prompt/discretizer differs from pinned official source")
        serialized = tools.Serializer().serialize(scaler.discretize(example))
        expected = scaler.inverse_discretize(tools.Serializer().inverse_serialize(serialized))
        actual, valid = codec.parse(serialized, inverse_scale, inverse_offset, 0.)
        np.testing.assert_allclose(actual[valid], expected[valid], rtol=1e-6, atol=1e-6)
    data, settings = data_for(config)
    physical, _ = build_graph(data.series, settings)
    with np.load(output/"correlation_graph.npz", allow_pickle=False) as source:
        correlation = source["adjacency"]
    seed()
    native = LlamaForCausalLM.from_pretrained(config["weights"]["directory"], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").requires_grad_(False).eval()
    model = ChatTimeGOFA(native, physical[:3, :3], correlation[:3, :3])
    batch = data.batch("train", [0])
    origin = int(batch["origins"][0])
    raw = data.series.values[origin-12:origin, :3][None]
    observed = data.series.mask[origin-12:origin, :3][None]
    encoded = codec.encode(raw, observed, data.mean.numpy()[:3])
    mechanisms = check_model(model, encoded, batch["graph_features"][:, :3])
    parameters = counts(model)
    assert parameters["trainable_real_dof"] == 1120914
    full_tokens = codec.encode(data.series.values[origin-12:origin][None],
        data.series.mask[origin-12:origin][None], data.mean.numpy())
    cache_estimate = len(data.windows["train"])*full_tokens.input_ids.numel()*(4096*2+1)
    assert cache_estimate > config["cache"]["disk_limit_gib"]*2**30
    assert not torch.cuda.is_initialized()
    runtime = {"python": sys.version, "executable": sys.executable, "packages": packages,
        "modules": {name: {"path": module.__file__, "sha256": digest(module.__file__)}
                    for name, module in (("torch", torch), ("transformers", transformers),
                                         ("numpy", np), ("sklearn", sklearn))},
        "critical_files": {module.__file__: digest(module.__file__)
                           for module in (llama_source, mask_source, generation_source, tokenizer_source)}}
    report = {"passed": True, "code_files": code_files(), "runtime": runtime,
        "cpu_test_cases": len(cases), "stage16_cases": len(chat_cases),
        "optional_skipped": sum(c.find("skipped") is not None for c in cases),
        "official_codec_examples": len(examples), "real_model_mechanisms": mechanisms,
        "real_model_nodes": 3, "parameters": parameters, "estimated_full_cache_bytes": cache_estimate,
        "cuda_initialized": False, "scope": "CPU verification only; no GPU accuracy or mechanism claim"}
    atomic_json(output/"cpu_preflight.json", report)
    print(json.dumps({"passed": True, "cases": len(cases), "stage16_cases": len(chat_cases),
                      "parameters": parameters["trainable_real_dof"], "cuda_initialized": False}), flush=True)


if __name__ == "__main__":
    main()
