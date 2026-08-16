
__all__ = ["PhysioEncoder", "SliceEncoder"]


def __getattr__(name: str):
    if name == "PhysioEncoder":
        from .physio_encoder import PhysioEncoder

        return PhysioEncoder
    if name == "SliceEncoder":
        from .slice_encoder import SliceEncoder

        return SliceEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
