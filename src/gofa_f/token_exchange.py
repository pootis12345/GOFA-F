"""Stage14 isolates data-patch and REG routes through the frozen backbone."""
from gofa_f.temporal_context import SketchData, TemporalSketchBolt


class TokenExchangeBolt(TemporalSketchBolt):
    def __init__(self, native, physical, correlation, horizon, variant, **architecture):
        if variant not in ("m1_data_only", "m2_reg_only") or architecture.get("output_adapter_rank") != 4:
            raise ValueError("Unregistered Stage14 token policy")
        if not native.chronos_config.use_reg_token:
            raise ValueError("Token role study requires the original REG token")
        super().__init__(native, physical, correlation, horizon, "t2_correlated", **architecture)
        self.token_policy = variant
        self.graph_architecture.pop("stage8_variant")
        self.graph_architecture.update({"stage14_variant": variant, "graph_token_policy": variant})

    def exchange(self, index, hidden, mask, features):
        graph_mask = mask.clone()
        available = mask[..., :-1].any(-1)
        if self.token_policy == "m1_data_only":
            graph_mask[..., -1] = False
        else:
            graph_mask[..., :-1] = False
            graph_mask[..., -1] = available
        # Only graph messages use this mask. Native attention retains both original token roles.
        return self.graph[str(index)](hidden, self.adjacency, graph_mask, features)
