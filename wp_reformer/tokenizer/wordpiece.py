import argparse
from tokenizers import BertWordPieceTokenizer
import json

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_file", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--vocab_size", type=int, default=20000)
parser.add_argument("--limit_alphabet", type=int, default=6000)

args = parser.parse_args()

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False,
    lowercase=False,
    wordpieces_prefix="##"
)

tokenizer.train(
    files=[args.corpus_file],
    limit_alphabet=args.limit_alphabet,
    vocab_size=args.vocab_size
)

tokenizer.save(args.output_file, True)

with open(args.output_file) as json_file:
    json_data = json.load(json_file)
    vocab_file = args.output_file + '.txt'
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for item in json_data["model"]["vocab"].keys():
            f.write(item + '\n')