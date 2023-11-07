import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil,wandb
from typing import Dict, List, Tuple
from sources.early_stopping import Early_Stopping
from sources.make_vocab import Make_vocab
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sources.electra import Electra
from mongo_processor import Mongo
from fastprogress.fastprogress import master_bar, progress_bar

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
    AutoModel,TFElectraModel,ElectraModel,BertTokenizer,
    AutoModelForPreTraining,AutoModelForMaskedLM
)
def tie_weights(generator, discriminator):
    generator.electra.embeddings.word_embeddings = discriminator.electra.embeddings.word_embeddings
    generator.electra.embeddings.position_embeddings = discriminator.electra.embeddings.position_embeddings
    generator.electra.embeddings.token_type_embeddings = discriminator.electra.embeddings.token_type_embeddings

class LogitsAdapter(torch.nn.Module):
    def __init__(self, adaptee):
        super().__init__()
        self.adaptee = adaptee

    def forward(self, *args, **kwargs):
        return self.adaptee(*args, **kwargs)[0]
def run(do_train,args):
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.target_gpu))
        logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
        logger.info('We will use the GPU:{}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        logger.info('No GPU available, using the CPU instead.')
    if args.model_type == 'electra':
        vocab_file = Make_vocab().make_vocab(args)
        tokenizer = ElectraTokenizer(vocab_file = vocab_file)
    elif args.model_type == 'electra_multi':
        tokenizer=AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    print("VOCAB SIZE:",len(list(tokenizer.get_vocab().keys())))
    if do_train:

        generator = AutoModelForMaskedLM.from_pretrained("google/electra-base-generator")
        discriminator = AutoModelForPreTraining.from_pretrained("google/electra-base-discriminator")
        
        if args.model_type == 'electra':
            config = AutoConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")
        elif args.model_type == 'electra_multi':
            config = AutoConfig.from_pretrained("bert-base-multilingual-cased")
            
        tie_weights(generator, discriminator)
        model = Electra(
                config = config,
                generator=LogitsAdapter(generator),
                discriminator=LogitsAdapter(discriminator),
                num_tokens=tokenizer.vocab_size,
                mask_ignore_token_ids=[tokenizer.vocab['[CLS]'], tokenizer.vocab['[SEP]']],
                random_token_prob=0.0).to(device)
        train_dataset= load_and_cache_examples(args, tokenizer, evaluate=False) 
        global_step, tr_loss = train(args, train_dataset, model, tokenizer,device,generator,discriminator)
    else:
        generator = AutoModelForMaskedLM.from_pretrained("google/electra-base-generator")
        discriminator = AutoModelForPreTraining.from_pretrained(args.load_ck)
        discriminator2 = AutoModelForPreTraining.from_pretrained(args.load_ck)
        tie_weights(generator, discriminator)
        model = Electra(
                LogitsAdapter(generator),
                LogitsAdapter(discriminator),
                num_tokens=tokenizer.vocab_size,
                mask_ignore_token_ids=[tokenizer.vocab['[CLS]'], tokenizer.vocab['[SEP]']],
                random_token_prob=0.0).to(device)        
        eval_dataset= load_and_cache_examples(args, tokenizer, evaluate=False)
        global_step, eval_loss = evaluate(args, eval_dataset, model, tokenizer,device,generator,discriminator,discriminator2)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, block_size=512):
        if args.op == 'train':
            mongo = Mongo(mongo_uri="mongodb://admin:qwer1234@192.168.0.40:27017/NER-refine",
                db_name = 'NER-refine',
                collection = 'DAPT_data_TOTLA')
        else:
            mongo = Mongo(mongo_uri="mongodb://admin:qwer1234@192.168.0.40:27017/NER-refine",
                db_name = 'NER-refine',
                collection = 'DAPT_data_TEST')
        datas = mongo.find_item()
        lines = []
        for i in datas:
            sentence = i['sentence']
            lines.append(sentence)
        self.lines = lines
        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)
  
def load_and_cache_examples(args, tokenizer, evaluate=False):
    return TextDataset(tokenizer, args, block_size=args.block_size)

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels

def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,device,generator,discriminator) -> Tuple[int, float]:
    wandb.init(project="DAPT", entity="ayaan")
    wandb.watch(model)
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size 
    
    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, collate_fn=collate
    )
    print(len(train_dataloader))
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    early_stopping = Early_Stopping(verbose = True)
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = master_bar(range(args.num_train_epochs))
    for epoch in train_iterator:
        epoch_iterator = progress_bar(train_dataloader, parent=train_iterator)
        logger.info('####################### Epoch {}: ELECTRA Pre-Training Start #######################'.format(epoch))
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)

        for step, batch in enumerate(epoch_iterator):
            inputs, labels = mask_tokens(batch, tokenizer, args) 
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.train()
            loss, mlm_loss, disc_loss, gen_acc, disc_acc, disc_labels, disc_predictions = model(inputs)
            loss.backward()

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()  
            model.zero_grad()
            global_step += 1
            train_iterator.write('------------------------------------------------[ELECTRA Train] Epoch {}/{}, Steps {}/{}------------------------'.format(epoch, args.num_train_epochs, step, len(train_dataloader)))
            train_iterator.write('                                    train loss {}'.format(loss))
            ckpt ="TRAIN_{}epoch_{}step_loss{}.bin".format(epoch,step,loss)
            ck_path = os.path.join(args.ck_path, ckpt)
            if epoch > 0 and epoch <10  and args.save_steps > 0 and global_step % args.save_steps == 0:
                early_stopping(loss,model, ck_path,discriminator)
                wandb.log({"(TRAIN)loss": round(int(loss),2), "epoch":epoch})
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()





def evaluate(args, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,device,generator,discriminator,discriminator2) -> Tuple[int, float]:
    """ Evaluate the model """
    args.eval_batch_size = args.per_gpu_train_batch_size #* max(1, args.n_gpu)
    # def tie_weights(generator, discriminator):
    #     generator.electra.embeddings.word_embeddings = discriminator.electra.embeddings.word_embeddings
    #     generator.electra.embeddings.position_embeddings = discriminator.electra.embeddings.position_embeddings
    #     generator.electra.embeddings.token_type_embeddings = discriminator.electra.embeddings.token_type_embeddings
    
    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    #train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, collate_fn=collate
    )
    print(len(eval_dataloader))

    model.eval()
    # Evaluate!
    logger.info("***** Running Evaluate *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total evaluate batch size (w. parallel, distributed & accumulation) = %d",
        args.eval_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )

    early_stopping = Early_Stopping(verbose = True)
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    eval_loss, logging_loss = 0.0, 0.0

    epoch_iterator = progress_bar(eval_dataloader)
    fff = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            inputs, labels = mask_tokens(batch, tokenizer, args) 
            inputs = inputs.to(device)
            labels = labels.to(device)

            loss, loss_mlm, loss_disc, acc_gen, acc_disc, disc_labels, disc_pred = model(inputs)
            final_hidden_states = discriminator(inputs)
            f =[]
            #f.append(sen)
            with open("./em", "a") as f:
                f.write("{}\n".format(final_hidden_states))
            fff.append(f)
            eval_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            #discriminator.electra.save_pretrained(f'{args.output_dir}/ckpt/{step}')
            # optimizer.step()
            # scheduler.step()  
            # model.zero_grad()
            global_step += 1
            logger.info('------------------------------------------------[Eval] Steps {}/{}------------------------'.format(step, len(eval_dataloader)))
            #logger.info('                                    eval loss {}'.format(loss))
            # ckpt ="TRAIN_{}epoch_{}step_loss{}.pt".format(epoch,step,loss)
            # ck_path = os.path.join(args.output_dir, ckpt)
            # if args.save_steps > 0 and global_step % args.save_steps == 0:
            #     early_stopping(loss,model, ck_path,discriminator)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        # if args.max_steps > 0 and global_step > args.max_steps:
        #     train_iterator.close()
        #     break

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()
    return global_step, eval_loss/global_step

