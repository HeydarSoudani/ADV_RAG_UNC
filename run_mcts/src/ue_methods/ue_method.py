import numpy as np
from typing import Union
from abc import ABC, abstractmethod
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast


class UEMethod(ABC):    
    def __init__(self):
        # self.normalizer = SigmoidNormalizer(threshold = 0, std = 1.0)#default dummy normalizer
        pass

    def __str__(self):
        return f"{self.__class__.__name__} with {str(self.__dict__)}"
    
    @abstractmethod
    def __call__(self, model, tokenizer, context, question, generated_text):
        raise NotImplementedError("Subclasses must implement this method")