"""Registered CPU/CUDA mechanism checks; no test targets and no tolerance retries."""
from dataclasses import replace
import tempfile

import torch

from gofa_f.chattime import PrefixLRU


def assert_tree_equal(first, second):
    if torch.is_tensor(first):
        assert torch.equal(first.cpu(), second.cpu())
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_tree_equal(a, b)
    else:
        assert first == second


def check_model(model, encoded, features, check=lambda: None, gpu=False, binding="cpu-check"):
    from gofa_f.chattime_task import (gradients, initial_state, load_checkpoint,
        make_optimizer, optimizer_update, save_checkpoint)
    device = encoded.input_ids.device
    # These tolerances are registered before any CUDA run. State restoration itself
    # remains exact; subsequent CUDA BF16 trajectories are checked numerically.
    tolerance = {"rtol": 2e-3, "atol": 2e-3} if gpu or model.native.dtype == torch.bfloat16 else {
                 "rtol": 1e-6, "atol": 1e-6}
    native_versions = {name: p._version for name, p in model.native.named_parameters()}
    text_ids = encoded.input_ids[:1, :2]
    text_mask = torch.ones_like(text_ids)
    check()
    with torch.no_grad():
        text_before = model.native(input_ids=text_ids, attention_mask=text_mask, use_cache=False).logits
        # Same batch shape, so this checks the exact manual layer/mask/position path.
        reference = model.native.model(input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask, use_cache=False).last_hidden_state
        prefix = model.prefix(encoded, node_chunk=len(encoded.input_ids), check=check)
        positions = torch.arange(encoded.attention_mask.shape[1], device=device)[None].expand_as(encoded.attention_mask)
        last = positions.masked_fill(~encoded.attention_mask, -1).max(1).values
        wanted = reference[torch.arange(len(last), device=device), last].float()
        actual = model.last_hidden(prefix, encoded, features)
        torch.testing.assert_close(actual, wanted, **tolerance)
        initial = model.from_prefix(prefix, encoded, features)
        torch.testing.assert_close(initial, encoded.last[:, None].expand_as(initial), rtol=0, atol=0)
    del reference, wanted, actual

    with torch.no_grad():
        generator = torch.Generator(device="cpu").manual_seed(43)
        model.head.weight.copy_(torch.randn(model.head.weight.shape, generator=generator).to(device)*1e-4)
        for module in model.graph.values():
            module.gate.fill_(.1)
    results, grads = [], []
    for cached in (False, True):
        check()
        model.zero_grad(set_to_none=True)
        prediction = model.from_prefix(prefix, encoded, features) if cached else model(
            encoded, features, node_chunk=len(encoded.input_ids), check=check)
        prediction.square().mean().backward()
        results.append(prediction.detach())
        grads.append(gradients(model))
    torch.testing.assert_close(results[0], results[1], **tolerance)
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        assert torch.isfinite(grads[0][name]).all(), name
        torch.testing.assert_close(grads[0][name], grads[1][name], **tolerance)
    graph_active = any(float(value.abs().sum()) > 0 for name, value in grads[0].items() if name.startswith("graph."))
    assert graph_active
    assert all(p.grad is None for p in model.native.parameters())
    shared_native_prefill = False
    if bool(encoded.attention_mask.all()):
        check()
        with torch.no_grad():
            generated = model.native.generate(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask,
                do_sample=False, temperature=1., top_p=1., max_new_tokens=1, use_cache=True,
                pad_token_id=model.native.config.eos_token_id, return_dict_in_generate=True,
                output_hidden_states=True)
            native_prefix = (generated.hidden_states[0][model.graph_layers[0]+1], encoded.attention_mask)
        model.zero_grad(set_to_none=True)
        shared_prediction = model.from_prefix(native_prefix, encoded, features)
        shared_prediction.square().mean().backward()
        torch.testing.assert_close(shared_prediction, results[0], **tolerance)
        shared_gradients = gradients(model)
        assert shared_gradients.keys() == grads[0].keys()
        for name in shared_gradients:
            torch.testing.assert_close(shared_gradients[name], grads[0][name], **tolerance)
        shared_native_prefill = True
        del generated, native_prefix, shared_prediction, shared_gradients

    # Missing nodes cannot send or receive a graph delta, including all-empty neighborhoods.
    with torch.no_grad():
        missing = replace(encoded, present=torch.zeros_like(encoded.present))
        empty = model.from_prefix(prefix, missing, features)
        torch.testing.assert_close(empty, encoded.last[:, None].expand_as(empty), rtol=0, atol=0)
        module = next(iter(model.graph.values()))
        width = model.native.config.hidden_size
        n = encoded.shape[1]
        example = torch.randn(2, n, 1, width, device=device, dtype=next(model.native.parameters()).dtype)
        mask = torch.ones(2, n, 1, device=device, dtype=torch.bool)
        two_features = features[:1].expand(2, -1, -1).clone()
        first = module(example, model.physical, mask, two_features)
        changed = example.clone()
        changed[1] += 5
        second = module(changed, model.physical, mask, two_features)
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)

    # An interruption inside an effective batch preserves pending gradient sums.
    optimizer = make_optimizer(model)
    state = initial_state()
    model.zero_grad(set_to_none=True)
    prediction = model.from_prefix(prefix, encoded, features)
    prediction.abs().sum().backward()
    state.update(pending_count=prediction.numel(), pending_windows=1, pending_loss=float(prediction.detach().abs().sum()))
    with tempfile.TemporaryDirectory() as directory:
        save_checkpoint(directory, model, optimizer, state, binding)
        pending = gradients(model)
        original = model.trainable_state()
        optimizer_update(model, optimizer, state)
        expected_weights = model.trainable_state()
        expected_optimizer = optimizer.state_dict()
        restored = load_checkpoint(directory, model, optimizer, binding)
        assert_tree_equal(original, model.trainable_state())
        assert_tree_equal(pending, gradients(model))
        assert restored["pending_count"] == prediction.numel() and restored["pending_windows"] == 1
        optimizer_update(model, optimizer, restored)
        assert_tree_equal(expected_weights, model.trainable_state())
        assert_tree_equal(expected_optimizer, optimizer.state_dict())
    with torch.no_grad():
        text_after = model.native(input_ids=text_ids, attention_mask=text_mask, use_cache=False).logits
    torch.testing.assert_close(text_before, text_after, rtol=0, atol=0)
    assert native_versions == {name: p._version for name, p in model.native.named_parameters()}
    assert all(p.grad is None for p in model.native.parameters())
    cache = PrefixLRU(binding, max_bytes=1024)
    key = cache.key(encoded)
    changed = replace(encoded, input_ids=encoded.input_ids.clone())
    changed.input_ids[0, 0] = (changed.input_ids[0, 0]+1) % model.native.config.vocab_size
    assert key != cache.key(changed) and key != PrefixLRU(binding+"changed").key(encoded)
    return {"initial_persistence": True, "zero_graph_native_hidden": True,
        "cached_prediction_and_gradients": True, "graph_gradient_active": graph_active,
        "batch_graph_isolation": True, "empty_nodes": True, "pending_gradient_resume_exact": True,
        "optimizer_resume_exact": True, "native_weights_unchanged": True, "text_bypass_exact": True,
        "native_prefill_shared_prediction_and_gradient": shared_native_prefill,
        "numeric_tolerance": tolerance, "checkpoint_state_tolerance": "exact"}
