from tokenizers import Tokenizer, normalizers, trainers, models, processors, decoders
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import PreTokenizer, Whitespace
from my_pretokenizer import PreKoTok
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_file", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--vocab_size", type=int, default=20000)

args = parser.parse_args()

ko_tok_pretokenizer = PreTokenizer.custom(PreKoTok())

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.BertNormalizer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False,
    lowercase=False,
)

tokenizer.pre_tokenizer = ko_tok_pretokenizer

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=args.vocab_size, special_tokens=special_tokens)

tokenizer.model = models.WordPiece(unk_token="[UNK]")
tokenizer.train([args.corpus_file], trainer=trainer)

cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")

tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)

tokenizer.decoder = decoders.WordPiece(prefix="##")

tokenizer.pre_tokenizer = Whitespace()
tokenizer.save(args.output_file)
