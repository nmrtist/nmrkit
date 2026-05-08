from pathlib import Path

import pytest

import nmrkit as nk


def test_delta_reader_accepts_hypercomplex_indirect_storage():
    path = Path(__file__).parents[2] / "demo" / "gibberellic_acid_hsqc_edited_dec-1.jdf"
    if not path.exists():
        pytest.skip("JEOL Delta demo file is not available")

    data = nk.read(str(path))

    assert data.shape == (1024, 512)
    assert data.dimensions[1].size == 512
    assert data.dimensions[1].domain_metadata["logical_size"] == 256
    assert data.dimensions[1].domain_metadata["complex_pair_encoding"] == "separated"
