from typing import Union


class TextFrontend():
    def __init__(
        self, 
        process_type: Union[None, str]
    ):
        if process_type is None:
            self.processor = lambda x: x.strip("\n").strip(" ")
        else:
            raise NotImplementedError(f"Not supported: {process_type}")
    
    def __call__(self, text:str):
        return self.processor(text)