import pytest
import pandas as pd
from easyTrainer.data.text import TextualPreparator

@pytest.fixture
def sample_data():
    return pd.DataFrame({"col1": ["Hello WORLD 123 !!!", "A new EXAMPLE hére"]})

def test_to_lower(sample_data):
    tp = TextualPreparator(to_lower=1)
    result = tp.prepare(sample_data["col1"]).rename(None)
    expected = pd.Series(["hello world 123 !!!", "a new example hére"])
    pd.testing.assert_series_equal(result, expected)

def test_to_upper(sample_data):
    tp = TextualPreparator(to_upper=1)
    result = tp.prepare(sample_data["col1"]).rename(None)
    expected = pd.Series(["HELLO WORLD 123 !!!", "A NEW EXAMPLE HÉRE"])
    pd.testing.assert_series_equal(result, expected)

def test_drop_big_spaces(sample_data):
    tp = TextualPreparator(drop_big_spaces=1)
    result = tp.prepare(sample_data["col1"]).rename(None)
    expected = pd.Series(["Hello WORLD 123 !!!", "A new EXAMPLE hére"])
    pd.testing.assert_series_equal(result, expected)

def test_drop_digits(sample_data):
    tp = TextualPreparator(drop_digits=1)
    result = tp.prepare(sample_data["col1"]).rename(None)
    expected = pd.Series(["Hello WORLD   !!!", "A new EXAMPLE hére"])
    pd.testing.assert_series_equal(result, expected)

def test_drop_special_characters(sample_data):
    tp = TextualPreparator(drop_special_characters=1)
    result = tp.prepare(sample_data["col1"]).rename(None)
    expected = pd.Series(["Hello WORLD 123    ", "A new EXAMPLE hére"])
    pd.testing.assert_series_equal(result, expected)

def test_drop_accents(sample_data):
    tp = TextualPreparator(drop_accents=1)
    result = tp.prepare(sample_data["col1"]).rename(None)
    expected = pd.Series(["Hello WORLD 123 !!!", "A new EXAMPLE here"])
    pd.testing.assert_series_equal(result, expected)

def test_drop_words_less_than_N_letters(sample_data):
    tp = TextualPreparator(drop_words_less_than_N_letters=(1, 4, True))
    result = tp.prepare(sample_data["col1"]).rename(None)
    expected = pd.Series(["Hello WORLD 123 !!!", "EXAMPLE hére"])
    pd.testing.assert_series_equal(result, expected)

def test_combined_transformations(sample_data):
    tp = TextualPreparator(
        to_lower=1,
        drop_digits=2,
        drop_special_characters=3,
        drop_accents=4,
        drop_words_less_than_N_letters=(5, 4, True)
    )
    result = tp.prepare(sample_data["col1"]).rename(None)
    expected = pd.Series(["hello world", "example here"])
    pd.testing.assert_series_equal(result, expected)

def test_custom_function(sample_data):
    def custom_reverse_text(txt):
        return txt[::-1]

    tp = TextualPreparator(custom_steps={"reverse_text": (1, custom_reverse_text)})
    result = tp.prepare(sample_data["col1"]).rename(None)
    expected = pd.Series(["!!! 321 DLROW olleH", "eréh ELPMAXE wen A"])
    pd.testing.assert_series_equal(result, expected)