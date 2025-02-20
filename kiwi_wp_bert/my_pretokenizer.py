from kiwipiepy import Kiwi

class PreKoTok:
    def __init__(self):
        self.kiwi = Kiwi(model_type="sbg")

    def split(self, _n, normalized):
        return [
            normalized[token.start:token.end]
            for token
            in self.kiwi.tokenize(str(normalized))
        ]

    def pre_tokenize(self, pretok):
        pretok.split(self.split)
