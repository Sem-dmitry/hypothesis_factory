"""Open CoScientist Agents - Multi-agent system for AI co-scientist research."""

__version__ = "0.0.1"

# Lazy re-exports (PEP 562). The framework pulls heavy optional dependencies
# (LangGraph, gpt-researcher, ...); importing them eagerly here would force every
# lightweight consumer (e.g. ``coscientist.model_factory``) to drag in the whole
# stack. We expose the public names lazily so they only load when accessed.
_LAZY_EXPORTS = {
    "CoscientistConfig": "coscientist.framework",
    "CoscientistFramework": "coscientist.framework",
    "CoscientistState": "coscientist.global_state",
    "CoscientistStateManager": "coscientist.global_state",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
