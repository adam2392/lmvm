"""Dataset readers honouring the SPEC §6.2 dataset contract."""

from .clevr import CLEVRDataset, clevr_bbox_from_pixel_coords, clevr_question_type
from .common import vqa_collate_fn

__all__ = [
    "CLEVRDataset",
    "clevr_bbox_from_pixel_coords",
    "clevr_question_type",
    "vqa_collate_fn",
]
