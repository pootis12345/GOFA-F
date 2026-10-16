"""Stage15 combines the fixed REG-only message route and budgeted rank6 head."""
from gofa_f.head_capacity import HeadCapacityBolt, SketchData
from gofa_f.token_exchange import TokenExchangeBolt


class RegCapacityBolt(HeadCapacityBolt):
    def __init__(self, native, physical, correlation, horizon, variant, **architecture):
        if variant != "r1_reg_rank6" or not native.chronos_config.use_reg_token:
            raise ValueError("Unregistered Stage15 REG route")
        super().__init__(native, physical, correlation, horizon, "h1_rank6", **architecture)
        self.token_policy = "m2_reg_only"
        self.graph_architecture.pop("stage13_variant")
        self.graph_architecture.update({"stage15_variant": variant,
            "graph_token_policy": self.token_policy})

    exchange = TokenExchangeBolt.exchange
