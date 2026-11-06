"""Dataset loaders owned by Archana."""

from adr.data.loaders.arc import load_arc
from adr.data.loaders.mmlu import load_mmlu
from adr.data.loaders.opinionqa import load_opinionqa

__all__ = ["load_arc", "load_mmlu", "load_opinionqa"]
