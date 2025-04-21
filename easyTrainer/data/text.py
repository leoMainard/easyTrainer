import re
import unicodedata
import numpy as np
import pandas as pd

from typing import Optional, Union, List, Tuple, Dict, Callable
from types import FunctionType

from easyTrainer.data.base_preparator import BasePreparator
from easyTrainer.data.utils import extract_params, load_stopwords

class TextualPreparator(BasePreparator):
    """
    A class that applies multiple text transformations in a defined order.
    The arguments of this class are functions to be applied, with their order specified.
    The first argument of each function is its execution order in the text preprocessing pipeline.

    Args:
        to_lower (Optional[int]): 
            Order in which to apply lowercase transformation.
        
        to_upper (Optional[int]): 
            Order in which to apply uppercase transformation.
        
        drop_stopwords (Optional[int]):
            Order in which to remove stopwords.
        
        drop_digits (Optional[int]): 
            Order in which to remove digits.
        
        lemmatize (Optional[int]): 
            Order in which to apply lemmatization using spaCy.
        
        drop_special_characters (Optional[int]): 
            Order to remove special characters (punctuation etc.).
        
        drop_accents (Optional[int]): 
            Order in which to remove accents from characters.
        
        drop_words_less_than_N_letters (Optional[Union[int, Tuple[int, int, bool]]]): 
            Either the order (int), or a tuple (order, n_letters, isalpha_flag) indicating 
            the order, minimum number of letters, and whether to filter only alphabetic words.

    Example:
        >>> tp = TextualPreparator(
        >>>     to_lower=2, 
        >>>     drop_accents=3,
        >>>     drop_digits=1,
        >>>     drop_special_characters=4,
        >>>     drop_words_less_than_N_letters=(5, 3, True)
        >>> )
        >>> df = pd.DataFrame({"col1": ["Hello WORLD 123 !!!", "A new EXAMPLE hére"]})
        >>> df["result"] = tp.run(df["col1"])
        >>> print(df)
    """

    def __init__(
            self,
            to_lower: Optional[int] = None,
            to_upper: Optional[int] = None,
            drop_stopwords: Optional[int] = None,
            drop_big_spaces: Optional[int] = None,
            drop_digits: Optional[int] = None,
            lemmatize: Optional[int] = None,
            drop_special_characters: Optional[int] = None,
            drop_accents: Optional[int] = None,
            drop_words_less_than_N_letters: Optional[Union[int, Tuple[int, int, bool]]] = None,
            custom_steps: Optional[Dict[str, Tuple[int, Callable[[str], str]]]] = None
    ):
        """
        A class that applies multiple text transformations in a defined order.
        The arguments of this class are functions to be applied, with their order specified.
        The first argument of each function is its execution order in the text preprocessing pipeline.

        Args:
            to_lower (Optional[int]): 
                Order in which to apply lowercase transformation.
            
            to_upper (Optional[int]): 
                Order in which to apply uppercase transformation.
            
            drop_stopwords (Optional[int]):
                Order in which to remove stopwords.
            
            drop_big_spaces (Optional[int]):
                Order in which to remove multiple spaces.

            drop_digits (Optional[int]): 
                Order in which to remove digits.
            
            lemmatize (Optional[int]): 
                Order in which to apply lemmatization using spaCy.
            
            drop_special_characters (Optional[int]): 
                Order to remove special characters (punctuation etc.).
            
            drop_accents (Optional[int]): 
                Order in which to remove accents from characters.
            
            drop_words_less_than_N_letters (Optional[Union[int, Tuple[int, int, bool]]]): 
                Either the order (int), or a tuple (order, n_letters, isalpha_flag) indicating 
                the order, minimum number of letters, and whether to filter only alphabetic words.
            
            custom_steps (Optional[Dict[str, Tuple[int, Callable]]]):
                Dictionary of custom functions in the form
                {"function_name": (order, function)}.
                Each function must take a string as input and return a string.

        Example:
            >>> tp = TextualPreparator(
            >>>     to_lower=2, 
            >>>     drop_accents=3,
            >>>     drop_digits=1,
            >>>     drop_special_characters=4,
            >>>     drop_words_less_than_N_letters=(5, 3, True)
            >>> )
            >>> df = pd.DataFrame({"col1": ["Hello WORLD 123 !!!", "A new EXAMPLE hére"]})
            >>> df["result"] = tp.run(df["col1"])
            >>> print(df)
        """
        super().__init__()

        self.custom_functions = {}
        self.order = {
            "txt_to_upper": to_upper,
            "txt_to_lower": to_lower,
            "drop_stopwords": drop_stopwords,
            "drop_big_spaces": drop_big_spaces,
            "drop_digits": drop_digits,
            "lemmatize": lemmatize,
            "drop_special_characters": drop_special_characters,
            "drop_accents": drop_accents,
            "drop_words_less_than_N_letters": extract_params(drop_words_less_than_N_letters, (2, True))[0],
        }

        if custom_steps:
            if not isinstance(custom_steps, dict):
                raise TypeError("custom_steps must be a dictionary.")
            if not all(isinstance(k, str) for k in custom_steps.keys()):
                raise TypeError("All keys in custom_steps must be strings.")
            if not all(isinstance(v, tuple) and len(v) == 2 for v in custom_steps.values()):
                raise TypeError("All values in custom_steps must be tuples of (order, function).")
            if not all(isinstance(v[0], int) for v in custom_steps.values()):
                raise TypeError("The first element of each tuple in custom_steps must be an integer (order).")
            if not all(isinstance(v[1], FunctionType) for v in custom_steps.values()):
                raise TypeError("The second element of each tuple in custom_steps must be a callable function.")
            
            for name, (step_order, func) in custom_steps.items():
                self.custom_functions[name] = func
                self.order[name] = step_order

        self.order = {k: v for k, v in self.order.items() if v is not None}
        self.order = dict(sorted(self.order.items(), key=lambda item: item[1]))

        if "drop_stopwords" in self.order:
            self.stopwords = load_stopwords()

        if "drop_words_less_than_N_letters" in self.order:
            self.n_letters, self.n_letters_isalpha = extract_params(drop_words_less_than_N_letters, (2, True))[1:]

        self._nlp = None


    def txt_to_lower(self, txt: str) -> str:
        """ Convert text to lowercase """
        return txt.lower()

    def txt_to_upper(self, txt: str) -> str:
        """ Convert text to uppercase """
        return txt.upper()

    def drop_stopwords(self, txt: str) -> str:
        """ Removes stopwords from the text """
        if not self.stopwords:
            return txt
        words = txt.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        return " ".join(filtered_words)

    def drop_big_spaces(self, txt: str) -> str:
        """ Remove multiple spaces from the text """
        return re.sub(r'\s+', ' ', txt).strip()
    
    def drop_digits(self, txt: str) -> str:
        """ Remove all digits from the text """
        return re.sub(r'\d+', ' ', txt)

    def lemmatize(self, txt: str) -> str:
        """ Lemmatizes the text using spaCy's French model """
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("fr_core_news_sm")
            except ImportError:
                raise ImportError("Please install spaCy and the French model: `pip install spacy && python -m spacy download fr_core_news_sm`")
        doc = self._nlp(txt)
        return " ".join([token.lemma_ for token in doc])

    def drop_special_characters(self, txt: str) -> str:
        """ Remove all special characters from the text """
        return re.sub(r"[^\w\s]", " ", txt)

    def drop_accents(self, txt: str) -> str:
        """ Remove accents from all characters in the text """
        return ''.join(
            c for c in unicodedata.normalize('NFKD', txt) if unicodedata.category(c) != 'Mn'
        )

    def drop_words_less_than_N_letters(self, txt: str) -> str:
        """ Removes words shorter than a specified length """
        if not isinstance(self.n_letters, int) or self.n_letters < 1:
            raise ValueError("n_letters must be a positive integer.")
        if not isinstance(self.n_letters_isalpha, bool):
            raise ValueError("isalpha_flag must be a boolean.")
        words = txt.split()
        if self.n_letters_isalpha:
            words_filtered = [word for word in words if len(word) >= self.n_letters or not word.isalpha()]
        else:
            words_filtered = [word for word in words if len(word) >= self.n_letters]
        return " ".join(words_filtered)

    def _apply_transformations(self, text: str) -> str:
        """ Applies all transformations in the defined order to a single string """
        for func_name in self.order:
            if func_name in self.custom_functions:
                text = self.custom_functions[func_name](text)
            else:
                text = getattr(self, func_name)(text)
        return text

    def prepare(
        self, 
        data: Union[List[str], pd.Series, pd.DataFrame, np.ndarray], 
        all: bool = False,
        encoder_name_to_fit: Optional[str] = None,
        custom_encoder_to_fit: Optional[object] = None,
        custom_encoder_fit: Optional[object] = None,
    ) -> Union[List[str], pd.Series, pd.DataFrame, dict]:
        """
        Apply the configured text transformations to the provided data, 
        and optionally encode it using a textual encoder.

        Args:
            data (List[str] | pd.Series | pd.DataFrame | np.ndarray): 
                Text data to preprocess. If a numpy array is provided, it will be converted
                to a Series (1D) or DataFrame (2D).
            
            all (bool): 
                If True and data is a DataFrame, apply transformations to all columns (converted to str).
                If False, apply only to object/string columns.
            
            encoder_name_to_fit (str, optional): 
                If specified, encode the processed data using this method ('tfidf', 'word2vec').
            
            custom_encoder_to_fit (object, optional): 
                A user-provided encoder with a `fit_transform()` method.

            custom_encoder_fit (object, optional): 
                A user-provided fitted encoder with a `transform()` method.

        Returns:
            dict: 
                - If no encoder: a dict containing 'data' : the preprocessed data.
                - If encoder is specified: a dict containing 'data', 'encoded_data', and 'vectorizer'.

        Raises:
            TypeError: If the input is not a list, Series, or DataFrame.

        Notes:
            When encoding is used and the input is a DataFrame, the text of each row is concatenated
            into a single string (joining all columns), then encoded as a whole. 
            Each row of the encoded result represents one full line of the DataFrame.
        """
        if data is None or (isinstance(data, (pd.Series, pd.DataFrame)) and data.empty):
            raise ValueError("The provided data is empty.")
        
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.Series(data)
            elif data.ndim == 2:
                data = pd.DataFrame(data)
            else:
                raise TypeError("Only 1D or 2D numpy arrays are supported.")
        
        if isinstance(data, pd.Series):
            processed = data.astype(str).apply(self._apply_transformations)

        elif isinstance(data, list):
            processed = [self._apply_transformations(str(x)) for x in data]

        elif isinstance(data, pd.DataFrame):
            if all:
                processed = data.applymap(lambda x: self._apply_transformations(str(x)))
            else:
                df_copy = data.copy()
                for col in df_copy.select_dtypes(include="object").columns:
                    df_copy[col] = df_copy[col].astype(str).apply(self._apply_transformations)
                processed = df_copy
        else:
            raise TypeError("The provided data type is not supported. Use list, pd.Series, or pd.DataFrame.")

        if encoder_name_to_fit or custom_encoder_to_fit or custom_encoder_fit:
            return TextualEncoder(
                processed, 
                encoder_name_to_fit= encoder_name_to_fit, 
                custom_encoder_to_fit= custom_encoder_to_fit,
                custom_encoder_fit= custom_encoder_fit
            ).get_results()

        return {"data": processed}



class TextualEncoder:
    """
    Encodes textual data using a specified encoding method (e.g., TF-IDF, Word2Vec).

    Attributes:
        data (Union[List[str], pd.Series, pd.DataFrame]): Textual data to encode.
        encoder_name (str): Name of the encoding method ('tfidf', 'word2vec').
        custom_encoder_to_fit (Optional[object]): User-provided encoder with a `fit_transform()` method.
        custom_encoder_fit (Optional[object]): User-provided fitted encoder with a `transform()` method.
        vectorizer (object): The encoder used (can be reused later).
        encoded_data (array-like): Encoded representation of the text data.
    """

    def __init__(
            self, 
            data: Union[List[str], pd.Series, pd.DataFrame], 
            encoder_name_to_fit: str = None, 
            custom_encoder_to_fit=None,
            custom_encoder_fit=None
        ):
        """
        Initialize and encode the data using the provided encoder.

        Args:
            data (List[str] | pd.Series | pd.DataFrame): Textual data to encode.
            encoder_name_to_fit (str): Encoding method name. Supported: 'tfidf', 'word2vec'.
            custom_encoder_to_fit (Optional): A user-defined encoder with a `fit_transform()` method.
            custom_encoder_fit (Optional): A user-defined fitted encoder with a `transform()` method.

        Raises:
            ValueError: If the encoder name is unknown or unsupported.

        Notes:
            Only one encoding method will be applied, based on the following priority:

            1. If `custom_encoder_fit` is provided, it will be used, assuming it is already fitted.
            The method `transform()` will be called on the data.

            2. Else if `custom_encoder_to_fit` is provided, it will be fitted and applied using
            `fit_transform()`.

            3. Else if `encoder_name_to_fit` is set to "tfidf", a `TfidfVectorizer` from scikit-learn
            will be created and used to encode the data.

            4. If `encoder_name_to_fit` is "word2vec", it is not yet implemented and will raise
            a `NotImplementedError`.

            5. If none of the above apply, a `ValueError` is raised for an unknown or missing
            encoding method.
        """

        if data is None or (isinstance(data, (pd.Series, pd.DataFrame)) and data.empty):
            raise ValueError("The provided data is empty.")
        
        if not isinstance(data, (list, pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError("Data must be a list, Series, or DataFrame or ndarray.")
        
        if encoder_name_to_fit is not None and not isinstance(encoder_name_to_fit, str):
            raise TypeError("Encoder name must be a string or None.")
        
        if custom_encoder_to_fit is not None and not hasattr(custom_encoder_to_fit, "fit_transform"):
            raise ValueError("Custom encoder must have a 'fit_transform' method.")
        
        if custom_encoder_fit is not None and not hasattr(custom_encoder_fit, "transform"):
            raise ValueError("Custom fitted encoder must have a 'transform' method.")

        self.encoder_name_to_fit = encoder_name_to_fit
        self.custom_encoder_to_fit = custom_encoder_to_fit

        # Preprocess text data into a single column
        if isinstance(data, pd.DataFrame):
            self.data = data.astype(str).agg(" ".join, axis=1)
        elif isinstance(data, pd.Series):
            self.data = data.astype(str)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                self.data = pd.Series(data.astype(str))
            elif data.ndim == 2:
                # Join all columns per row if 2D
                self.data = pd.Series([" ".join(map(str, row)) for row in data])
            else:
                raise ValueError("Only 1D or 2D arrays are supported.")
        elif isinstance(data, list):
            self.data = list(map(str, data))
        else:
            raise TypeError("Unsupported data type.")

        if custom_encoder_fit is not None:
            self.vectorizer = custom_encoder_fit
            self.encoded_data = custom_encoder_fit.transform(self.data)

        elif custom_encoder_to_fit is not None:
            self.vectorizer = custom_encoder_to_fit
            self.encoded_data = custom_encoder_to_fit.fit_transform(self.data)

        elif self.encoder_name_to_fit.lower() == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer()
            self.encoded_data = self.vectorizer.fit_transform(self.data)

        elif self.encoder_name_to_fit.lower() == "word2vec":
            raise NotImplementedError("Word2Vec encoding is not yet implemented.")

        else:
            raise ValueError(f"Unknown encoding method: {encoder_name_to_fit}")

    def get_results(self) -> dict:
        """
        Return the encoded data and the encoder used.

        Returns:
            dict: {
                'data': original textual data (after flattening if needed),
                'encoded_data': encoded data,
                'vectorizer': encoder used
            }
        """
        return {
            "data": self.data,
            "encoded_data": self.encoded_data,
            "vectorizer": self.vectorizer,
        }
