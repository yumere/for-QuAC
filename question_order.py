from __future__ import absolute_import, division, print_function

import argparse
import logging
import os

import torch
from pytorch_transformers import AdamW
from pytorch_transformers import BertTokenizer
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from question_order_util import QuestionDataset

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class BertOrderPrediction(BertPreTrainedModel):
    def __init__(self, config):
        super(BertOrderPrediction, self).__init__(config)
        self.max_que_length = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        hidden_size = 1000
        self.encoder = nn.LSTM(input_size=config.hidden_size, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        self.output_layer = nn.Linear(hidden_size * 2, self.max_que_length)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None,
                q_len=None, q_no=None, q_mask=None):
        sequence_outputs, pooled_outputs = self.bert(input_ids, token_type_ids, attention_mask, position_ids, head_mask)
        rnn_inputs = pack_sequence(pooled_outputs.split(q_len), enforce_sorted=False)
        encoder_outputs, (h_n, _) = self.encoder(rnn_inputs)
        encoder_outputs = pad_packed_sequence(encoder_outputs, batch_first=True)[0]
        outputs = self.output_layer(encoder_outputs)
        return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=None, type=str, required=True)
    parser.add_argument("--predict_file", default=None, type=str, required=True)
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--max_que_length", default=15, type=int)
    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--do_eval", action="store_true", default=False)
    parser.add_argument("--do_lower_case", action="store_true", default=True)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int)
    parser.add_argument("--per_gpu_dev_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--logging_steps", default=10, type=int)
    parser.add_argument("--saving_steps", default=100, type=int)
    parser.add_argument("--no_cuda", default=False, action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.dev_batch_size = args.per_gpu_dev_batch_size * max(1, args.n_gpu)

    if args.do_train:
        model = BertOrderPrediction.from_pretrained('bert-base-uncased', num_labels=args.max_que_length)
        model.zero_grad()
        model.to(device)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        dataset = QuestionDataset(args.train_file, tokenizer=tokenizer, max_seq_length=64, max_que_length=args.max_que_length)
        loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=10, drop_last=True, collate_fn=QuestionDataset.collate_fn)
        args.t_total = len(loader) // args.num_train_epochs
        max_grad_norm = 1.0

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.t_total)

        tb_writer = SummaryWriter(logdir=args.output_dir)

        global_step = 0
        for e in tqdm(list(range(args.num_train_epochs)), desc="Epoch", ncols=75):
            for i, batch in enumerate(tqdm(loader, desc="Steps", ncols=75)):
                model.train()

                global_step += 1
                tokens = batch[0]

                inputs = {
                    "input_ids": batch[1].to(device),
                    "attention_mask": batch[2].to(device),
                    "token_type_ids": batch[3].to(device),
                    "q_len": batch[5],
                    "q_no": batch[6].to(device),
                    "q_mask": batch[7].to(device)
                }

                outputs = model(**inputs)
                loss = criterion(outputs.reshape(-1, args.max_que_length), inputs["q_no"].reshape(-1))
                loss.backward()
                clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # scheduler.step()
                optimizer.step()
                model.zero_grad()

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tqdm.write("Step: {:,} Loss: {}".format(global_step, loss.item()))
                    # tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", loss.item(), global_step)

                if args.saving_steps > 0 and global_step % args.saving_steps == 0:
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(args.output_dir)
                    torch.save("", os.path.join(args.output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", args.output_dir)

        tb_writer.close()

    if args.do_eval:
        model = BertOrderPrediction.from_pretrained(args.output_dir)
        model.to(device)

        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        dataset = QuestionDataset(args.predict_file, tokenizer=tokenizer, max_seq_length=64, max_que_length=args.max_que_length)
        loader = DataLoader(dataset, batch_size=args.dev_batch_size, shuffle=False, num_workers=10, drop_last=True, collate_fn=QuestionDataset.collate_fn)

        for i, batch in enumerate(loader):
            model.eval()

            with torch.no_grad():
                tokens = batch[0]

                inputs = {
                    "input_ids": batch[1].to(device),
                    "attention_mask": batch[2].to(device),
                    "token_type_ids": batch[3].to(device),
                    "q_len": batch[5],
                    "q_no": batch[6].to(device),
                    "q_mask": batch[7].to(device)
                }

                outputs = model(**inputs)
                print(outputs)
