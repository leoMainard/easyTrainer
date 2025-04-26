from easytrainer.data.utils import extract_params, load_stopwords

from unittest.mock import mock_open, patch, MagicMock


# ----------- extract_params -----------

def test_extract_params_with_tuple():
    assert extract_params(("a", 1)) == ("a", 1)

def test_extract_params_with_single_value():
    assert extract_params("a") == ("a", None)

def test_extract_params_with_single_value_and_default():
    assert extract_params("a", default=42) == ("a", 42)


# ----------- load_stopwords -----------

def test_load_stopwords_file_found():
    fake_file_content = "le\nla\nles"
    mock_file = mock_open(read_data=fake_file_content)

    mock_resource = MagicMock()
    mock_resource.open = mock_file

    mock_joinpath = MagicMock(return_value=mock_resource)
    mock_files = MagicMock(return_value=MagicMock(joinpath=mock_joinpath))

    with patch("easytrainer.data.utils.files", mock_files):
        result = load_stopwords()

    assert result == ["le", "la", "les"]

def test_load_stopwords_file_not_found():
    mock_resource = MagicMock()
    mock_resource.open.side_effect = FileNotFoundError

    mock_joinpath = MagicMock(return_value=mock_resource)
    mock_files = MagicMock(return_value=MagicMock(joinpath=mock_joinpath))

    with patch("easytrainer.data.utils.files", mock_files):
        result = load_stopwords()

    assert result == set()

def test_load_stopwords_file_empty():
    fake_file_content = ""
    mock_file = mock_open(read_data=fake_file_content)

    mock_resource = MagicMock()
    mock_resource.open = mock_file

    mock_joinpath = MagicMock(return_value=mock_resource)
    mock_files = MagicMock(return_value=MagicMock(joinpath=mock_joinpath))

    with patch("easytrainer.data.utils.files", mock_files):
        result = load_stopwords()

    assert result == set()

def test_load_stopwords_file_with_empty_lines():
    fake_file_content = "\nle\n\nla\nles\n"
    mock_file = mock_open(read_data=fake_file_content)

    mock_resource = MagicMock()
    mock_resource.open = mock_file

    mock_joinpath = MagicMock(return_value=mock_resource)
    mock_files = MagicMock(return_value=MagicMock(joinpath=mock_joinpath))

    with patch("easytrainer.data.utils.files", mock_files):
        result = load_stopwords()

    assert result == ["le", "la", "les"]