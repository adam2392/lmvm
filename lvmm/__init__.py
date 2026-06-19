"""LVMM proof-of-concept package.

Modules
-------
filter_bank : Data-adaptive Random Fourier Feature Vision Core (fixed, no learned params).
database    : External Visual Knowledge Database (FAISS-backed prototype store).
injection   : Entity-region prototype injection (LVMM only).
model       : Shared Transformer ReasoningModel (LVMM and Baseline).
datasets    : CLEVR / GQA dataset readers honouring the dataset contract (SPEC §6.2).
vocab       : Question / answer vocabularies.
"""

from .filter_bank import DataAdaptiveRFF
from .database import VisualKnowledgeDB
from .injection import inject_prototypes, bbox_to_token_indices, pool_region
from .model import ReasoningModel
from .vocab import Vocab

__all__ = [
    "DataAdaptiveRFF",
    "VisualKnowledgeDB",
    "inject_prototypes",
    "bbox_to_token_indices",
    "pool_region",
    "ReasoningModel",
    "Vocab",
]
