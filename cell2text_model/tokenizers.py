import logging
from typing import List, Dict, Union
from transformers import AutoTokenizer
from Geneformer.geneformer.tokenizer import TranscriptomeTokenizer


logger = logging.getLogger(__name__)

class TextTokenizer:
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, texts: List[str], max_length: int = 512) -> Dict[str, Union[List[int], List[List[int]]]]:
        encoded = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True
        )
        return encoded


class CellTokenizer:
    def __init__(self, vocab_file: str = None):
        # You may need to load a custom tokenizer or vocab here
        self.vocab_file = vocab_file
        # Initialize your tokenizer logic (e.g., load vocab, define encoding rules)

    def tokenize(self, tokens: List[str], max_length: int = 2048) -> Dict[str, Union[List[int], List[List[int]]]]:
        """
        This should convert gene tokens (e.g. ENSG ids or symbols) into indices.
        """
        # Stub implementation — replace with your logic
        token_ids = [self._token_to_id(t) for t in tokens]
        return {
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
        }

    def _token_to_id(self, token: str) -> int:
        """
        Convert a single token to its ID. You should customize this with your own vocab logic.
        """
        raise NotImplementedError("Define your gene token vocabulary or mapping here.")



