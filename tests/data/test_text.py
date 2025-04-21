import pytest
import pandas as pd
from easyTrainer.data.text import TextualPreparator, TextualEncoder
from sklearn.feature_extraction.text import CountVectorizer

@pytest.fixture
def sample_data():
    return pd.DataFrame({"col1": ["Hello WORLD 123 !!!", "A new EXAMPLE hére"]})


# ----- TextualPreparator Tests -----

def test_prepare_with_dataframe(sample_data):
    tp = TextualPreparator()
    result = tp.prepare(sample_data)
    assert isinstance(result, dict)
    assert "data" in result
    assert len(result["data"]) == len(sample_data)

def test_prepare_with_series(sample_data):
    tp = TextualPreparator()
    result = tp.prepare(sample_data["col1"])
    assert isinstance(result, dict)
    assert "data" in result
    assert len(result["data"]) == len(sample_data["col1"])

def test_prepare_with_list(sample_data):
    tp = TextualPreparator()
    result = tp.prepare(sample_data["col1"].tolist())
    assert isinstance(result, dict)
    assert "data" in result
    assert len(result["data"]) == len(sample_data["col1"])

def test_prepare_with_numpy_array(sample_data):
    tp = TextualPreparator()
    result = tp.prepare(sample_data["col1"].values)
    assert isinstance(result, dict)
    assert "data" in result
    assert len(result["data"]) == len(sample_data["col1"])

def test_prepare_no_data():
    tp = TextualPreparator()
    with pytest.raises(ValueError):
        tp.prepare(data=None)

def test_prepare_no_encoder(sample_data):
    tp = TextualPreparator()
    result = tp.prepare(sample_data["col1"])
    assert isinstance(result, dict)
    assert "data" in result
    assert len(result["data"]) == len(sample_data["col1"])

def test_prepare_with_encoder(sample_data):
    tp = TextualPreparator()
    result = tp.prepare(sample_data["col1"], custom_encoder_to_fit=CountVectorizer())
    assert isinstance(result, dict)
    assert "data" in result
    assert len(result["data"]) == len(sample_data["col1"])
    assert "encoded_data" in result
    assert result["encoded_data"].shape[0] == len(sample_data["col1"])
    assert "vectorizer" in result
    assert isinstance(result["vectorizer"], CountVectorizer)

def test_invalid_encoder_name(sample_data):
    tp = TextualPreparator()
    with pytest.raises(ValueError):
        tp.prepare(sample_data["col1"], encoder_name_to_fit="unknown_encoder")

def test_to_lower(sample_data):
    tp = TextualPreparator(to_lower=1)
    result = tp.prepare(sample_data["col1"])["data"].rename(None)
    expected = pd.Series(["hello world 123 !!!", "a new example hére"])
    pd.testing.assert_series_equal(result, expected)

def test_to_upper(sample_data):
    tp = TextualPreparator(to_upper=1)
    result = tp.prepare(sample_data["col1"])["data"].rename(None)
    expected = pd.Series(["HELLO WORLD 123 !!!", "A NEW EXAMPLE HÉRE"])
    pd.testing.assert_series_equal(result, expected)

def test_drop_big_spaces(sample_data):
    tp = TextualPreparator(drop_big_spaces=1)
    result = tp.prepare(sample_data["col1"])["data"].rename(None)
    expected = pd.Series(["Hello WORLD 123 !!!", "A new EXAMPLE hére"])
    pd.testing.assert_series_equal(result, expected)

def test_drop_digits(sample_data):
    tp = TextualPreparator(drop_digits=1)
    result = tp.prepare(sample_data["col1"])["data"].rename(None)
    expected = pd.Series(["Hello WORLD   !!!", "A new EXAMPLE hére"])
    pd.testing.assert_series_equal(result, expected)

def test_drop_special_characters(sample_data):
    tp = TextualPreparator(drop_special_characters=1)
    result = tp.prepare(sample_data["col1"])["data"].rename(None)
    expected = pd.Series(["Hello WORLD 123    ", "A new EXAMPLE hére"])
    pd.testing.assert_series_equal(result, expected)

def test_drop_accents(sample_data):
    tp = TextualPreparator(drop_accents=1)
    result = tp.prepare(sample_data["col1"])["data"].rename(None)
    expected = pd.Series(["Hello WORLD 123 !!!", "A new EXAMPLE here"])
    pd.testing.assert_series_equal(result, expected)

def test_drop_words_less_than_N_letters(sample_data):
    tp = TextualPreparator(drop_words_less_than_N_letters=(1, 4, True))
    result = tp.prepare(sample_data["col1"])["data"].rename(None)
    expected = pd.Series(["Hello WORLD 123 !!!", "EXAMPLE hére"])
    pd.testing.assert_series_equal(result, expected)

def test_invalid_drop_words_less_than_N_letters(sample_data):
    tp = TextualPreparator(drop_words_less_than_N_letters=(1, "4", True))
    with pytest.raises(ValueError):
        tp.prepare(sample_data["col1"])

def test_combined_transformations(sample_data):
    tp = TextualPreparator(
        to_lower=1,
        drop_digits=2,
        drop_special_characters=3,
        drop_accents=4,
        drop_words_less_than_N_letters=(5, 4, True)
    )
    result = tp.prepare(sample_data["col1"])["data"].rename(None)
    expected = pd.Series(["hello world", "example here"])
    pd.testing.assert_series_equal(result, expected)

def test_custom_function(sample_data):
    def custom_reverse_text(txt):
        return txt[::-1]

    tp = TextualPreparator(custom_steps={"reverse_text": (1, custom_reverse_text)})
    result = tp.prepare(sample_data["col1"])["data"].rename(None)
    expected = pd.Series(["!!! 321 DLROW olleH", "eréh ELPMAXE wen A"])
    pd.testing.assert_series_equal(result, expected)

def test_invalid_custom_function(sample_data):
    def invalid_function(txt):
        return txt + 1  # Invalid operation

    with pytest.raises(TypeError):
        tp = TextualPreparator(custom_steps={"invalid_function": (1, invalid_function)})
        tp.prepare(sample_data["col1"])

def test_invalid_custom_steps():
    with pytest.raises(TypeError):
        TextualPreparator(custom_steps={"invalid_step": (1, "not_a_function")})

def test_invalid_custom_steps_type():
    with pytest.raises(TypeError):
        TextualPreparator(custom_steps={"invalid_step": (1, 123)})

def test_invalid_custom_steps_order_type():
    with pytest.raises(TypeError):
        TextualPreparator(custom_steps={"invalid_step": (1.5, lambda x: x)})


# ----- TextualEncoder Tests -----

def test_encoder_with_dataframe(sample_data):
    tp = TextualPreparator(to_lower=1)
    processed_data = tp.prepare(sample_data)["data"]
    encoder = TextualEncoder(processed_data, encoder_name_to_fit="tfidf")
    results = encoder.get_results()
    assert "data" in results
    assert "encoded_data" in results
    assert "vectorizer" in results
    assert results["encoded_data"].shape[0] == len(processed_data)

def test_encoder_with_series(sample_data):
    tp = TextualPreparator(to_lower=1)
    processed_data = tp.prepare(sample_data["col1"])["data"]
    encoder = TextualEncoder(processed_data, encoder_name_to_fit="tfidf")
    results = encoder.get_results()
    assert "data" in results
    assert "encoded_data" in results
    assert "vectorizer" in results
    assert results["encoded_data"].shape[0] == len(processed_data)

def test_encoder_with_list(sample_data):
    tp = TextualPreparator(to_lower=1)
    processed_data = tp.prepare(sample_data["col1"].tolist())["data"]
    encoder = TextualEncoder(processed_data, encoder_name_to_fit="tfidf")
    results = encoder.get_results()
    assert "data" in results
    assert "encoded_data" in results
    assert "vectorizer" in results
    assert results["encoded_data"].shape[0] == len(processed_data)

def test_encoder_with_numpy_array(sample_data):
    tp = TextualPreparator(to_lower=1)
    processed_data = tp.prepare(sample_data["col1"].values)["data"]
    encoder = TextualEncoder(processed_data, encoder_name_to_fit="tfidf")
    results = encoder.get_results()
    assert "data" in results
    assert "encoded_data" in results
    assert "vectorizer" in results
    assert results["encoded_data"].shape[0] == len(processed_data)

def test_tfidf_encoding(sample_data):
    tp = TextualPreparator(to_lower=1)
    processed_data = tp.prepare(sample_data["col1"])["data"]
    encoder = TextualEncoder(processed_data, encoder_name_to_fit="tfidf")
    results = encoder.get_results()
    assert "data" in results
    assert "encoded_data" in results
    assert "vectorizer" in results
    assert results["encoded_data"].shape[0] == len(processed_data)

def test_custom_encoder(sample_data):
    tp = TextualPreparator(to_lower=1)
    processed_data = tp.prepare(sample_data["col1"])["data"]
    custom_encoder = CountVectorizer()
    encoder = TextualEncoder(processed_data, encoder_name_to_fit="custom", custom_encoder_to_fit=custom_encoder)
    results = encoder.get_results()
    assert "data" in results
    assert "encoded_data" in results
    assert "vectorizer" in results
    assert results["encoded_data"].shape[0] == len(processed_data)

def test_invalid_encoder_name_to_fit(sample_data):
    tp = TextualPreparator(to_lower=1)
    processed_data = tp.prepare(sample_data["col1"])["data"]
    with pytest.raises(ValueError):
        TextualEncoder(processed_data, encoder_name_to_fit="unknown_encoder")

def test_word2vec_not_implemented(sample_data):
    tp = TextualPreparator(to_lower=1)
    processed_data = tp.prepare(sample_data["col1"])["data"]
    with pytest.raises(NotImplementedError):
        TextualEncoder(processed_data, encoder_name_to_fit="word2vec")

def test_invalid_custom_encoder_to_fit(sample_data):
    tp = TextualPreparator(to_lower=1)
    processed_data = tp.prepare(sample_data["col1"])["data"]
    invalid_encoder = "not_a_valid_encoder"
    with pytest.raises(ValueError):
        TextualEncoder(processed_data, custom_encoder_to_fit=invalid_encoder)

def test_custom_encoder_fit(sample_data):
    tp = TextualPreparator(to_lower=1)
    processed_data = tp.prepare(sample_data["col1"])["data"]
    custom_encoder = CountVectorizer()
    encoder = TextualEncoder(processed_data, custom_encoder_to_fit=custom_encoder)
    results_with_encoder_to_fit = encoder.get_results()

    fitted_vectorizer = results_with_encoder_to_fit["vectorizer"]
    encoder2 = TextualEncoder(processed_data, custom_encoder_fit=fitted_vectorizer)
    results_with_fitted_encoder = encoder2.get_results()
    assert "data" in results_with_fitted_encoder
    assert "encoded_data" in results_with_fitted_encoder
    assert "vectorizer" in results_with_fitted_encoder
    assert results_with_fitted_encoder["encoded_data"].shape[0] == len(processed_data)
    
def test_invalid_custom_encoder_fit(sample_data):
    tp = TextualPreparator(to_lower=1)
    processed_data = tp.prepare(sample_data["col1"])["data"]
    invalid_encoder = "not_a_valid_encoder"
    with pytest.raises(ValueError):
        TextualEncoder(processed_data, custom_encoder_fit=invalid_encoder)

def test_empty_data_encoding():
    with pytest.raises(ValueError):
        TextualEncoder([], encoder_name_to_fit="tfidf")
