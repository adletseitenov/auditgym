"""AuditGym: a reproducible RL environment for a financial-audit agent."""

__version__ = "0.1.0"
SCHEMA_VERSION = "1"


def __getattr__(name: str):
    # Lazy, so `python -m auditgym serve` and the generator do not import the client.
    if name in ("Env", "EnvError"):
        from . import client

        return getattr(client, name)
    raise AttributeError(name)


__all__ = ["Env", "EnvError", "SCHEMA_VERSION", "__version__"]
