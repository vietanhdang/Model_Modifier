from typing import List, Dict, Any, Optional, Union
from abc import abstractmethod
import logging
from pathlib import Path
import langdetect
class BaseComponent:
    """
    A base class for implementing nodes in a Pipeline.
    """

    outgoing_edges: int
    subclasses: dict = {}
    pipeline_config: dict = {}

    def __init_subclass__(cls, **kwargs):
        """ This automatically keeps track of all available subclasses.
        Enables generic load() for all specific component implementations.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def get_subclass(cls, component_type: str):
        if component_type not in cls.subclasses.keys():
            raise Exception(f"Haystack component with the name '{component_type}' does not exist.")
        subclass = cls.subclasses[component_type]
        return subclass

    @classmethod
    def load_from_args(cls, component_type: str, **kwargs):
        """
        Load a component instance of the given type using the kwargs.
        
        :param component_type: name of the component class to load.
        :param kwargs: parameters to pass to the __init__() for the component. 
        """
        subclass = cls.get_subclass(component_type)
        instance = subclass(**kwargs)
        return instance

    @classmethod
    def load_from_pipeline_config(cls, pipeline_config: dict, component_name: str):
        """
        Load an individual component from a YAML config for Pipelines.
        :param pipeline_config: the Pipelines YAML config parsed as a dict.
        :param component_name: the name of the component to load.
        """
        if pipeline_config:
            all_component_configs = pipeline_config["components"]
            all_component_names = [comp["name"] for comp in all_component_configs]
            component_config = next(comp for comp in all_component_configs if comp["name"] == component_name)
            component_params = component_config["params"]

            for key, value in component_params.items():
                if value in all_component_names:  # check if the param value is a reference to another component
                    component_params[key] = cls.load_from_pipeline_config(pipeline_config, value)

            component_instance = cls.load_from_args(component_config["type"], **component_params)
        else:
            component_instance = cls.load_from_args(component_name)
        return component_instance

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any):
        """
        Method that will be executed when the node in the graph is called.
        The argument that are passed can vary between different types of nodes
        (e.g. retriever nodes expect different args than a reader node)
        See an example for an implementation in haystack/reader/base/BaseReader.py
        :param kwargs:
        :return:
        """
        pass

    def set_config(self, **kwargs):
        """
        Save the init parameters of a component that later can be used with exporting
        YAML configuration of a Pipeline.
        :param kwargs: all parameters passed to the __init__() of the Component.
        """
        if not self.pipeline_config:
            self.pipeline_config = {"params": {}, "type": type(self).__name__}
            for k, v in kwargs.items():
                if isinstance(v, BaseComponent):
                    self.pipeline_config["params"][k] = v.pipeline_config
                elif v is not None:
                    self.pipeline_config["params"][k] = v

class BaseConverter(BaseComponent):
    """
    Base class for implementing file converts to transform input documents to text format for ingestion in DocumentStore.
    """

    outgoing_edges = 1

    def __init__(
        self,
        remove_numeric_tables: bool = False,
        valid_languages: Optional[List[str]] = None,
    ):
        """
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        """

        # save init parameters to enable export of component config as YAML
        self.set_config(
            remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages
        )

        self.remove_numeric_tables = remove_numeric_tables
        self.valid_languages = valid_languages

    @abstractmethod
    def convert(
        self,
        file_path: Path,
        meta: Optional[Dict[str, str]],
        remove_numeric_tables: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
        encoding: Optional[str] = "utf-8",
    ) -> Dict[str, Any]:
        """
        Convert a file to a dictionary containing the text and any associated meta data.
        File converters may extract file meta like name or size. In addition to it, user
        supplied meta data like author, url, external IDs can be supplied as a dictionary.
        :param file_path: path of the file to convert
        :param meta: dictionary of meta data key-value pairs to append in the returned document.
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        :param encoding: Select the file encoding (default is `utf-8`)
        """
        pass

    def validate_language(self, text: str) -> bool:
        """
        Validate if the language of the text is one of valid languages.
        """
        if not self.valid_languages:
            return True

        try:
            lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = None

        if lang in self.valid_languages:
            return True
        else:
            return False

    def run(self, file_paths: Union[Path, List[Path]],  # type: ignore
            meta: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,  # type: ignore
            remove_numeric_tables: Optional[bool] = None,  # type: ignore
            valid_languages: Optional[List[str]] = None, **kwargs):  # type: ignore

        if isinstance(file_paths, Path):
            file_paths = [file_paths]

        if meta is None or isinstance(meta, dict):
            meta = [meta] * len(file_paths)  # type: ignore

        documents: list = []
        for file_path, file_meta in zip(file_paths, meta):
            documents.append(
                self.convert(
                    file_path=file_path,
                    meta=file_meta,
                    remove_numeric_tables=remove_numeric_tables,
                    valid_languages=valid_languages,
                )
            )

        result = {"documents": documents, **kwargs}
        return result, "output_1"
logger = logging.getLogger(__name__)
class TextConverter(BaseConverter):
    def convert(
        self,
        file_path: Path,
        meta: Optional[Dict[str, str]] = None,
        remove_numeric_tables: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
        encoding: Optional[str] = "utf-8",
    ) -> Dict[str, Any]:
        """
        Reads text from a txt file and executes optional preprocessing steps.
        :param file_path: path of the file to convert
        :param meta: dictionary of meta data key-value pairs to append in the returned document.
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        :param encoding: Select the file encoding (default is `utf-8`)
        :return: Dict of format {"text": "The text from file", "meta": meta}}
        """
        if remove_numeric_tables is None:
            remove_numeric_tables = self.remove_numeric_tables
        if valid_languages is None:
            valid_languages = self.valid_languages

        with open(file_path, encoding=encoding, errors="ignore") as f:
            text = f.read()
            pages = text.split("\f")

        cleaned_pages = []
        for page in pages:
            lines = page.splitlines()
            cleaned_lines = []
            for line in lines:
                words = line.split()
                digits = [word for word in words if any(i.isdigit() for i in word)]

                # remove lines having > 40% of words as digits AND not ending with a period(.)
                if remove_numeric_tables:
                    if words and len(digits) / len(words) > 0.4 and not line.strip().endswith("."):
                        logger.debug(f"Removing line '{line}' from {file_path}")
                        continue

                cleaned_lines.append(line)

            page = "\n".join(cleaned_lines)
            cleaned_pages.append(page)

        if valid_languages:
            document_text = "".join(cleaned_pages)
            if not self.validate_language(document_text):
                logger.warning(
                    f"The language for {file_path} is not one of {self.valid_languages}. The file may not have "
                    f"been decoded in the correct text format."
                )

        text = "".join(cleaned_pages)
        document = {"text": text, "meta": meta}
        return document

class BasePreProcessor(BaseComponent):
    outgoing_edges = 1

    def process(
        self,
        documents: Union[dict, List[dict]],
        clean_whitespace: Optional[bool] = True,
        clean_header_footer: Optional[bool] = False,
        clean_empty_lines: Optional[bool] = True,
        split_by: Optional[str] = "word",
        split_length: Optional[int] = 1000,
        split_overlap: Optional[int] = None,
        split_respect_sentence_boundary: Optional[bool] = True,
    ) -> List[dict]:
        """
        Perform document cleaning and splitting. Takes a single document as input and returns a list of documents.
        """
        raise NotImplementedError

    def clean(
        self, document: dict, clean_whitespace: bool, clean_header_footer: bool, clean_empty_lines: bool,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def split(
        self,
        document: dict,
        split_by: str,
        split_length: int,
        split_overlap: int,
        split_respect_sentence_boundary: bool,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def run(  # type: ignore
        self,
        documents: Union[dict, List[dict]],
        clean_whitespace: Optional[bool] = None,
        clean_header_footer: Optional[bool] = None,
        clean_empty_lines: Optional[bool] = None,
        split_by: Optional[str] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
        **kwargs,
    ):
        documents = self.process(
            documents=documents,
            clean_whitespace=clean_whitespace,
            clean_header_footer=clean_header_footer,
            clean_empty_lines=clean_empty_lines,
            split_by=split_by,
            split_length=split_length,
            split_overlap=split_overlap,
            split_respect_sentence_boundary=split_respect_sentence_boundary,
        )
        result = {"documents": documents, **kwargs}
        return result, "output_1"

class TextConverter(BaseConverter):
    def convert(
        self,
        file_path: Path,
        meta: Optional[Dict[str, str]] = None,
        remove_numeric_tables: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
        encoding: Optional[str] = "utf-8",
    ) -> Dict[str, Any]:
        """
        Reads text from a txt file and executes optional preprocessing steps.
        :param file_path: path of the file to convert
        :param meta: dictionary of meta data key-value pairs to append in the returned document.
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        :param encoding: Select the file encoding (default is `utf-8`)
        :return: Dict of format {"text": "The text from file", "meta": meta}}
        """
        if remove_numeric_tables is None:
            remove_numeric_tables = self.remove_numeric_tables
        if valid_languages is None:
            valid_languages = self.valid_languages

        with open(file_path, encoding=encoding, errors="ignore") as f:
            text = f.read()
            pages = text.split("\f")

        cleaned_pages = []
        for page in pages:
            lines = page.splitlines()
            cleaned_lines = []
            for line in lines:
                words = line.split()
                digits = [word for word in words if any(i.isdigit() for i in word)]

                # remove lines having > 40% of words as digits AND not ending with a period(.)
                if remove_numeric_tables:
                    if words and len(digits) / len(words) > 0.4 and not line.strip().endswith("."):
                        logger.debug(f"Removing line '{line}' from {file_path}")
                        continue

                cleaned_lines.append(line)

            page = "\n".join(cleaned_lines)
            cleaned_pages.append(page)

        if valid_languages:
            document_text = "".join(cleaned_pages)
            if not self.validate_language(document_text):
                logger.warning(
                    f"The language for {file_path} is not one of {self.valid_languages}. The file may not have "
                    f"been decoded in the correct text format."
                )

        text = "".join(cleaned_pages)
        document = {"text": text, "meta": meta}
        return document