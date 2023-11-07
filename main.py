import argparse
import six, os, torch
from sources.electra import Electra
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
    DataCollatorForLanguageModeling
)
# from flask import Flask
# from flask_cors import CORS
# from flask_restful_swagger_2 import Api

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--model_type", type=str, default='electra',choices=['electra','electra_multi','bert'], help="The model architecture to be trained or fine-tuned.")

    parser.add_argument(
        "--block_size",
        default=-512,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens)."
    )
    parser.add_argument("--learning_rate", default=3e-5)
    parser.add_argument("--adam_epsilon", default=1e-8)
    parser.add_argument("--warmup_steps", default=0)
    parser.add_argument("--max_steps", default=999999999999999999)
    parser.add_argument("--save_steps", default=1)
    parser.add_argument('--load_ck', type=str, default=None, help='Write checkpoint path')

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument("--per_gpu_train_batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--n_gpu", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--num_train_epochs", default=999999999999999999, type=float, help="Total number of training epochs to perform."
    )
    # parser.add_argument(
    #     "--train_data_file", default="K_text7", type=str, help="The input training data file (a text file)."
    # )
    parser.add_argument("--ck_path", default="none", type=str, help="The input training data file (a text file).")
    parser.add_argument('--op', type=str, default='train', help='Choose operation')
    parser.add_argument('--target_gpu', type=str, default='0', choices=['0','1','2'], help='Choose target GPU')
    args = parser.parse_args()
    
    if isinstance(args.ck_path,six.string_types):
        args.ck_path = os.path.join("./checkpoints", args.ck_path)
        if not os.path.exists(args.ck_path):
            os.mkdir(args.ck_path)
            # os.system("cp ./koelectra-base-v3-discriminator/config.json {}/".format(args.output_dir))
            # os.system("cp ./koelectra-base-v3-discriminator/tokenizer_config.json {}/".format(args.output_dir))
            # os.system("cp ./koelectra-base-v3-discriminator/vocab.txt {}/".format(args.output_dir))
    # if isinstance(args.train_data_file,six.string_types):
    #     args.train_data_file = os.path.join("./datas", args.train_data_file)
    if isinstance(args.load_ck,six.string_types):
        args.load_ck = os.path.join("./checkpoints", args.load_ck)
  
    if args.op == 'train' and args.model_type == 'electra':
        from sources.pretrain_electra import run
        run(do_train=True, args = args)
    elif args.op == 'train' and args.model_type == 'electra_multi':
        from sources.pretrain_electra import run
        run(do_train=True, args = args)
    elif args.op == 'train' and args.model_type == 'bert':
        from sources.pretrain_bert import run
        run(do_train=True, args = args)
    # if args.op == 'test':
    #     if not os.path.isfile('config.json'):
    #         os.system("cp ./koelectra-base-v3-discriminator/config.json {}/".format(args.load_ck))
    #         os.system("cp ./koelectra-base-v3-discriminator/tokenizer_config.json {}/".format(args.load_ck))
    #         os.system("cp ./koelectra-base-v3-discriminator/vocab.txt {}/".format(args.load_ck))
    #     from sources.pretrain2 import run
    #     run(do_train=False, args = args)
