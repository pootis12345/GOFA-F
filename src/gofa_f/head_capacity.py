"""Stage13 tests private output capacity with the locked Stage8 graph architecture."""
from gofa_f.temporal_context import SketchData, TemporalSketchBolt


class HeadCapacityBolt(TemporalSketchBolt):
    def __init__(self, native, physical, correlation, horizon, variant, **architecture):
        if variant != "h1_rank6" or architecture.get("output_adapter_rank") != 6:
            raise ValueError("Unregistered Stage13 private output rank")
        super().__init__(native, physical, correlation, horizon, "t2_correlated", **architecture)
        self.graph_architecture.pop("stage8_variant")
        self.graph_architecture.update({"stage13_variant": variant,
            "output_adapter_budget": "rank6 is maximum integer within 10% of historical METR-LA UFG-A trainable DOF"})
