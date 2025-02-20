from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import PreTokenizer
from my_pretokenizer import PreKoTok
from transformers import PreTrainedTokenizerFast, LineByLineTextDataset, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import argparse
from util import ModelConfig, TrainConfig
from sklearn.model_selection import train_test_split
from datasets import Dataset

parser = argparse.ArgumentParser()

parser.add_argument("--vocab_path", type=str)
parser.add_argument("--dataset_path", type=str)

args = parser.parse_args()

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=Tokenizer.from_file(args.vocab_path),
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
wrapped_tokenizer._tokenizer.pre_tokenizer = PreTokenizer.custom(PreKoTok())

with open(args.dataset_path, 'r') as f:
    lines = f.readlines()
train_lines, test_lines = train_test_split(lines, test_size=0.1, random_state=42)

train_dataset = Dataset.from_dict({'text': train_lines})
test_dataset = Dataset.from_dict({'text': test_lines})

def tokenize_function(examples):
    return wrapped_tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns('text')
test_dataset = test_dataset.remove_columns('text')

train_dataset.set_format(type='torch')
test_dataset.set_format(type='torch')

model_config = ModelConfig(config_path='kiwi_wp_bert/config/bert_model.json').get_config()
config = BertConfig(
    attention_probs_dropout_prob = model_config.attention_probs_dropout_prob,
    hidden_act = model_config.hidden_act,
    hidden_dropout_prob = model_config.hidden_dropout_prob,
    hidden_size = model_config.hidden_size,
    initializer_range = model_config.initializer_range,
    intermediate_size = model_config.intermediate_size,
    max_position_embeddings = model_config.max_position_embeddings,
    num_attention_heads = model_config.num_attention_heads,
    num_hidden_layers = model_config.num_hidden_layers,
    type_vocab_size = model_config.type_vocab_size,
    vocab_size = model_config.vocab_size,
    layer_norm_eps = model_config.layer_norm_eps
)

model = BertForMaskedLM(config)

data_collator = DataCollatorForLanguageModeling(tokenizer = wrapped_tokenizer,
                                                mlm = True,
                                                mlm_probability = 0.15
)

train_config = TrainConfig('kiwi_wp_bert/config/training.json').get_config()
training_args = TrainingArguments(
    output_dir = train_config.output_dir,
    learning_rate = train_config.learning_rate,
    per_device_train_batch_size = train_config.per_device_train_batch_size,
    per_device_eval_batch_size = train_config.per_device_eval_batch_size,
    num_train_epochs = train_config.num_train_epochs,
    weight_decay = train_config.weight_decay,
    eval_strategy = train_config.eval_strategy,
    save_strategy = train_config.save_strategy,
    logging_strategy = train_config.logging_strategy,
    report_to = train_config.report_to,
    logging_dir = train_config.logging_dir
)

trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = train_dataset,
    eval_dataset = test_dataset
)

trainer.train()
trainer.save_model('kiwi_wp_bert/model/')
