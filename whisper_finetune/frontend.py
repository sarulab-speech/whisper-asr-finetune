from typing import Union

def pyopenjtalk_kana(text):
    import pyopenjtalk
    kanas = pyopenjtalk.g2p(text, kana=True)
    return kanas


class TextFrontend():
    def __init__(
        self, 
        process_type: Union[None, str]
    ):
        if process_type is None:
            self.processor = lambda x: x.strip("\n").strip(" ")
        elif process_type == "pyopenjtalk_kana":
            self.processor = pyopenjtalk_kana
        else:
            raise NotImplementedError(f"Not supported: {process_type}")
    
    def __call__(self, text:str):
        return self.processor(text)