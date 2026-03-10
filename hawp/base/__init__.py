import importlib

__all__ = ["_C", "utils", "to_device", "setup_logger", "MetricLogger", "save_config", "WireframeGraph"]


def __getattr__(name):
    if name == "_C":
        from .csrc import _C

        globals()["_C"] = _C
        return _C
    if name == "utils":
        module = importlib.import_module(f"{__name__}.utils")
        globals()["utils"] = module
        return module
    if name == "to_device":
        from .utils.comm import to_device

        globals()["to_device"] = to_device
        return to_device
    if name == "setup_logger":
        from .utils.logger import setup_logger

        globals()["setup_logger"] = setup_logger
        return setup_logger
    if name == "MetricLogger":
        from .utils.metric_logger import MetricLogger

        globals()["MetricLogger"] = MetricLogger
        return MetricLogger
    if name == "save_config":
        from .utils.miscellaneous import save_config

        globals()["save_config"] = save_config
        return save_config
    if name == "WireframeGraph":
        from .wireframe import WireframeGraph

        globals()["WireframeGraph"] = WireframeGraph
        return WireframeGraph
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")