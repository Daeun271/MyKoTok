from transformers import pipeline, BertForMaskedLM
import argparse
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import PreTokenizer
from my_tokenizer import PreKoTok
from transformers import PreTrainedTokenizerFast

parser = argparse.ArgumentParser()

parser.add_argument("--vocab_path", type=str)

args = parser.parse_args()

tokenizer = Tokenizer.from_file(args.vocab_path)
tokenizer.pre_tokenizer = PreTokenizer.custom(PreKoTok())

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

model = BertForMaskedLM.from_pretrained('kiwi_wp_bert/model/')

fill_mask = pipeline(
    "fill-mask",
    model = model,
    tokenizer = wrapped_tokenizer
)

print(fill_mask("안녕하세요. 저는 테스트하는 중입니다."))