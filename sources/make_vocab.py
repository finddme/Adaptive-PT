import os,re
import argparse
from tokenizers import BertWordPieceTokenizer
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    ElectraTokenizer,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    DataCollatorForLanguageModeling,
    AutoModelForPreTraining,
    AutoModel,TFElectraModel,ElectraModel
)
from langdetect import detect
class Make_vocab:
    def __init__(self):
        self.corpus_file = '/workspace/NER_data/E_text7'
        self.vocab_size = 32000
        self.limit_alphabet = 6000
        
    def make_vocab(self,args):
        if args.model_type == 'electra':
            vocab_file = './vocab/med-wordpiece-vocab-koelectra.txt'
            pretrained_vocab = "./koelectra-base-v3-discriminator/vocab.txt"
            tokenizer = AutoTokenizer.from_pretrained("./monologg/koelectra-base-v3-discriminator")

        elif args.model_type == 'bert':
            vocab_file = './vocab/med-wordpiece-vocab-bert2.txt'
            pretrained_vocab = "./biobert_diseases_ner/vocab.txt"
            tokenizer = AutoTokenizer.from_pretrained("./biobert_diseases_ner")

        en = re.compile('[a-zA-Z]')
        if not os.path.isfile(vocab_file):
            med_elec_tokens = []
            with open(pretrained_vocab, 'r',encoding='UTF8') as f:
                while True:
                    line = f.readline()
                    if "unused" not in line:
                        med_elec_tokens.append(line)
                    if not line:
                        break

            b_tokenizer = BertWordPieceTokenizer(
                vocab_file=None,
                clean_text=True,
                handle_chinese_chars=True,
                strip_accents=False, 
                lowercase=False,
                wordpieces_prefix="##"
            )
            b_tokenizer.train(
                files=[self.corpus_file],
                limit_alphabet=self.limit_alphabet,
                vocab_size=self.vocab_size
            )
            elec_tokens = []
            for i in list(b_tokenizer.get_vocab().keys()):
                if i not in med_elec_tokens and re.search(en, i) is not None:
                    med_elec_tokens.append("{}\n".format(i))
            with open(vocab_file, 'w',encoding='UTF8') as ff:
                for ii in med_elec_tokens[:34001]:
                    ff.write(ii)

        return vocab_file


