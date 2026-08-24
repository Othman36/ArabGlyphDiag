from arab_glyph_diag.metadata import ARABIC_LETTERS, DOTTED_CLASS_IDS


def test_default_metadata_shape():
    assert len(ARABIC_LETTERS) == 28
    assert len(DOTTED_CLASS_IDS) == 15
    assert ARABIC_LETTERS[0].name == "alef"
    assert ARABIC_LETTERS[27].name == "yeh"
