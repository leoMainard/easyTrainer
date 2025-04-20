import re
import unicodedata
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
            for name, (step_order, func) in custom_steps.items():
                if not isinstance(func, FunctionType):
                    raise TypeError(f"The custom step '{name}' must be a function.")
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

    def prepare(self, data: Union[List[str], pd.Series, pd.DataFrame], all: bool = False) -> Union[List[str], pd.Series, pd.DataFrame]:
        """
        Apply the configured text transformations to the provided data.

        Args:
            data (Union[List[str], pd.Series, pd.DataFrame]): 
                List, Series or DataFrame of strings to preprocess.
            all (bool): 
                If True and data is a DataFrame, apply transformations to all columns (converted to str).
                If False, apply only to object/string columns.

        Returns:
            Union[List[str], pd.Series, pd.DataFrame]: 
                Preprocessed text.

        Raises:
            TypeError: If the input is not a list, Series, or DataFrame.
        """
        if isinstance(data, pd.Series):
            return data.astype(str).apply(self._apply_transformations)

        elif isinstance(data, list):
            return [self._apply_transformations(str(x)) for x in data]

        elif isinstance(data, pd.DataFrame):
            if all:
                return data.applymap(lambda x: self._apply_transformations(str(x)))
            else:
                df_copy = data.copy()
                for col in df_copy.select_dtypes(include="object").columns:
                    df_copy[col] = df_copy[col].astype(str).apply(self._apply_transformations)
                return df_copy

        else:
            raise TypeError("The provided data type is not supported. Use list, pd.Series, or pd.DataFrame.")



