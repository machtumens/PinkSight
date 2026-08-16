
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

_UNI = Path("data/pathology/uni/pytorch_model.bin")
_CONCH = Path("data/pathology/conch/pytorch_model.bin")
_TITAN = Path("data/pathology/features/TCGA_TITAN_features.pkl")
_TCGA = Path("data/genomics/tcga/brca_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt")
_METABRIC = Path("data/genomics/metabric/brca_metabric/data_mrna_illumina_microarray.txt")

_TITAN_DIM = 768  


def _load_state_dict(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return sd.get("state_dict", sd) if isinstance(sd, dict) else sd


@pytest.mark.skipif(not _UNI.exists(), reason="UNI weights not pulled (data/ gitignored)")
def test_uni_is_a_valid_vit_state_dict():
    sd = _load_state_dict(_UNI)
    assert "pos_embed" in sd and "cls_token" in sd, "not a ViT state dict"
    assert len(sd) > 100, "suspiciously few tensors — likely an LFS stub"


@pytest.mark.skipif(not _CONCH.exists(), reason="CONCH weights not pulled (data/ gitignored)")
def test_conch_is_a_valid_vlm_state_dict():
    sd = _load_state_dict(_CONCH)
    assert "logit_scale" in sd, "not a CLIP-style vision-language state dict"
    assert len(sd) > 100, "suspiciously few tensors — likely an LFS stub"


@pytest.mark.skipif(not _TITAN.exists(), reason="TITAN features not pulled (data/ gitignored)")
def test_titan_features_are_slide_vectors():
    import pickle

    with _TITAN.open("rb") as f:
        obj = pickle.load(f)
    assert set(obj) >= {"embeddings", "filenames"}, "unexpected TITAN pkl schema"
    emb, names = obj["embeddings"], obj["filenames"]
    assert emb.shape[1] == _TITAN_DIM, f"expected {_TITAN_DIM}-dim slide vectors, got {emb.shape}"
    assert emb.shape[0] == len(names), "embeddings/filenames length mismatch"


@pytest.mark.parametrize(
    ("path", "min_samples"),
    [
        pytest.param(_TCGA, 1000, id="tcga_rsem"),
        pytest.param(_METABRIC, 1900, id="metabric"),
    ],
)
def test_expression_matrix_is_genes_by_samples(path, min_samples):
    if not path.exists():
        pytest.skip(f"{path.name} not pulled (data/ gitignored)")
    head = pd.read_csv(path, sep="\t", nrows=5)  
    assert "Hugo_Symbol" in head.columns, "expression matrix not Hugo-keyed"
    n_samples = len(head.columns) - 2  
    assert n_samples >= min_samples, f"only {n_samples} sample columns — truncated download?"
