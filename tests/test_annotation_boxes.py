
from pathlib import Path

import numpy as np
import pytest

from pinksight.data.annotation_boxes import Box, crop, load_boxes

_FILE = Path("data/raw/Annotation_Boxes.xlsx")


def test_crop_extents_and_margin():
    vol = np.arange(20 * 20 * 20).reshape(20, 20, 20)
    box = Box("p", (5, 10), (6, 9), (2, 8))
    assert crop(vol, box).shape == (5, 3, 6)  
    assert crop(vol, box, margin=2).shape == (9, 7, 10)


def test_crop_margin_clipped_at_edges():
    vol = np.zeros((10, 10, 10))
    box = Box("p", (0, 10), (0, 10), (0, 10))  
    assert crop(vol, box, margin=5).shape == (10, 10, 10)  


@pytest.mark.skipif(not _FILE.exists(), reason="Annotation_Boxes.xlsx not present (data/ gitignored)")
def test_real_boxes_load():
    boxes = load_boxes()
    assert len(boxes) == 922  
    b1 = boxes["Breast_MRI_001"]
    assert b1.row == (234, 271) and b1.col == (308, 341) and b1.slice == (89, 112)
    for p in ("Breast_MRI_042", "Breast_MRI_043", "Breast_MRI_044"):
        assert p in boxes
