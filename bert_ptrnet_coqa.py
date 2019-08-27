from __future__ import absolute_import, division, print_function

import argparse
import logging
import math
import os

import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers import BertTokenizer
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert_ptrnet_coqa_util import CoQAOrderDataset

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class OrderNet(BertPreTrainedModel):
    def __init__(self, config):
        super(OrderNet, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        mlp_hidden_size = 1024
        self.mlp_hidden_size = mlp_hidden_size
        self.read = nn.Sequential(nn.Linear(config.hidden_size, mlp_hidden_size), GeLU(),
                                  nn.Linear(mlp_hidden_size, mlp_hidden_size), GeLU(),
                                  nn.Linear(mlp_hidden_size, mlp_hidden_size), GeLU())
        rnn_hidden_size = mlp_hidden_size
        self.proc_step = 5
        self.encoder = nn.LSTMCell(mlp_hidden_size, rnn_hidden_size)
        self.encoder_attn = nn.MultiheadAttention(embed_dim=rnn_hidden_size, num_heads=1, dropout=config.attention_probs_dropout_prob)
        self.proj = nn.Linear(mlp_hidden_size + rnn_hidden_size, rnn_hidden_size, bias=False)

        self.decoder = nn.LSTMCell(mlp_hidden_size, rnn_hidden_size)
        self.decoder_attn = nn.MultiheadAttention(embed_dim=rnn_hidden_size, num_heads=1, dropout=config.attention_probs_dropout_prob)

        self.apply(self.init_weights)

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor,
                segment_ids: torch.Tensor, question_mask: torch.Tensor):
        device = input_ids.device
        batch_size, max_q_len, seq_len = input_ids.shape
        q_len = question_mask.sum(dim=1)  # batch_size
        mask = question_mask.unsqueeze(-1).expand(-1, -1, seq_len)  # batch_size x max_q_len x  seq_len

        input_ids = input_ids.masked_select(mask == 1).reshape(-1, seq_len)
        input_mask = input_mask.masked_select(mask == 1).reshape(-1, seq_len)
        segment_ids = segment_ids.masked_select(mask == 1).reshape(-1, seq_len)

        sequence_outputs, pooled_outputs = self.bert(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

        memory = self.read(pooled_outputs)
        memory = pad_sequence(memory.split(q_len.tolist()))  # max_q_len, batch_size, hidden_size
        _, _, input_size = memory.shape
        init_x = torch.zeros(batch_size, input_size).to(device)
        h_t, c_t = [torch.zeros(batch_size, self.encoder.hidden_size).to(device) for i in range(2)]
        for i in range(self.proc_step):
            h_t, c_t = self.encoder(init_x, (h_t, c_t))
            attn_output, attn_output_weights = self.encoder_attn(h_t.unsqueeze(0), memory, memory, question_mask == 0)
            attn_output = attn_output.squeeze(0)
            h_t = self.proj(torch.cat([h_t, attn_output], dim=1))

        outputs = []
        for i in range(max_q_len):
            h_t, c_t = self.decoder(init_x, (h_t, c_t))
            attn_output, attn_output_weights = self.decoder_attn(h_t.unsqueeze(0), memory, memory, question_mask == 0)
            outputs.append(attn_output_weights.squeeze(1))

        probs = torch.stack(outputs, dim=1)
        return probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=None, type=str, required=True)
    parser.add_argument("--dev_file", default=None, type=str, required=True)
    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--do_eval", action="store_true", default=False)
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str)
    parser.add_argument("--output_dir", default=None, type=str, required=True)

    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    # TODO: Need to apply gradient accumulation
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int)
    parser.add_argument("--per_gpu_dev_batch_size", default=8, type=int)

    # dataset configuration
    parser.add_argument("--max_question_len", type=int, default=15, metavar="15")
    parser.add_argument("--max_sequence_len", type=int, default=24, metavar="24")
    parser.add_argument("--samples_no", type=int, default=5, metavar="5")

    parser.add_argument("--do_lower_case", action="store_true", default=False)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    parser.add_argument("--logging_steps", default=10, type=int)
    parser.add_argument("--saving_steps", default=100, type=int)
    parser.add_argument("--no_cuda", default=False, action="store_true")

    args = parser.parse_args()

    assert args.do_train or args.do_eval, "You must do train or eval by using --do_train/do_eval"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logger.warning("Device: {}, n_gpu: {}".format(device, args.n_gpu))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.dev_batch_size = args.per_gpu_dev_batch_size * max(1, args.n_gpu)

    if args.do_train:
        model = OrderNet.from_pretrained(args.model_name_or_path)
        model.zero_grad()
        model.to(device)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        dataset = CoQAOrderDataset(args.train_file, "coqa-train.pkl", args.do_lower_case,
                                   max_question_len=args.max_question_len, max_sequence_len=args.max_sequence_len,
                                   samples_no=args.samples_no)

        # TODO: Change shuffle state from False to True
        loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False, drop_last=True, num_workers=1, collate_fn=CoQAOrderDataset.collate_fn)
        args.t_total = len(loader) * args.num_train_epochs
        logger.info("Total step: {:,}".format(args.t_total))
        max_grad_norm = 1.0

        criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        no_decay = ["bias", "LayerNorm.weight"]

        # TODO: Check whether named_parameters return my mlp and lstm cell parameter
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.t_total)

        tb_writer = SummaryWriter(logdir=args.output_dir)

        global_step = 0
        for e in tqdm(list(range(args.num_train_epochs)), desc="Epoch", ncols=75):
            for i, batch in enumerate(tqdm(loader, desc="Step", ncols=75)):
                model.train()
                model.zero_grad()

                global_step += 1
                batch_size, max_q_len, max_seq_len = batch[0].shape
                inputs = {
                    "input_ids": batch[0].to(device),
                    "input_mask": batch[1].to(device),
                    "segment_ids": batch[2].to(device),
                    "question_mask": batch[4].to(device)
                }
                targets = batch[3].to(device)

                outputs = model(**inputs)
                loss = criterion(outputs.reshape(-1, max_q_len), targets.reshape(-1))
                loss = loss.sum() / batch_size
                loss.backward()
                clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scheduler.step()
                optimizer.step()

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tqdm.write("Step: {:,} Loss: {}".format(global_step, loss.item()))
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", loss.item(), global_step)

                if args.saving_steps > 0 and global_step % args.saving_steps == 0:
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(args.output_dir)
                    torch.save("", os.path.join(args.output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", args.output_dir)

    # TODO: evaluate
    if args.do_eval:
        pass