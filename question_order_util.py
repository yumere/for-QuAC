from __future__ import absolute_import, division, print_function

import json
import logging
import os
from itertools import chain
from random import shuffle

import torch
from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class QuestionDataset(Dataset):
    def __init__(self, file_name: str, tokenizer: BertTokenizer, max_seq_length, max_que_length,
                 cls_token="[CLS]", sep_token="[SEP]", pad_token=0,
                 sequence_a_segment_id=0, cls_token_segment_id=0, pad_token_segment_id=0, mask_padding_with_zero=True):
        super(QuestionDataset, self).__init__()

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_que_length = max_que_length
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.sequence_a_segment_id = sequence_a_segment_id
        self.cls_token_segment_id = cls_token_segment_id
        self.pad_token_segment_id = pad_token_segment_id
        self.mask_padding_with_zero = mask_padding_with_zero

        if not os.path.exists(file_name):
            logger.error("{} not exists".format(file_name))

        self.dataset = json.load(open(file_name, "rt", encoding="utf-8"))['data']

    def __getitem__(self, item):
        questions = self.dataset[item]['paragraphs'][0]['qas']
        q_text = [q['question'] for q in questions]
        q_len = len(questions)

        assert q_len <= self.max_que_length
        q_no = list(range(q_len))

        q_doc_tokens = []
        q_char_to_word_offset = []

        for q in q_text:
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True

            for c in q:
                if self.is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            q_doc_tokens.append(doc_tokens)
            q_char_to_word_offset.append(char_to_word_offset)

        batch_tokens = []
        batch_input_ids = []
        batch_input_mask = []
        batch_segment_ids = []
        batch_p_mask = []

        for doc_tokens, char_to_word_offset in zip(q_doc_tokens, q_char_to_word_offset):
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []

            for i, token in enumerate(doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = self.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tokens = []
            segment_ids = []
            p_mask = []

            tokens.append(self.cls_token)
            segment_ids.append(self.cls_token_segment_id)
            p_mask.append(0)

            tokens += all_doc_tokens
            segment_ids += [self.sequence_a_segment_id] * len(all_doc_tokens)
            p_mask += [1] * len(all_doc_tokens)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if self.mask_padding_with_zero else 0] * len(input_ids)

            while len(input_ids) < self.max_seq_length:
                input_ids.append(self.pad_token)
                input_mask.append(0 if self.mask_padding_with_zero else 1)
                segment_ids.append(self.pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length

            batch_tokens.append(tokens)
            batch_input_ids.append(input_ids)
            batch_input_mask.append(input_mask)
            batch_segment_ids.append(segment_ids)
            batch_p_mask.append(p_mask)

        shuffle(q_no)
        batch_tokens = [batch_tokens[i] for i in q_no]
        batch_input_ids = [batch_input_ids[i] for i in q_no]
        batch_input_mask = [batch_input_mask[i] for i in q_no]
        batch_segment_ids = [batch_segment_ids[i] for i in q_no]
        batch_p_mask = [batch_p_mask[i] for i in q_no]
        return batch_tokens, batch_input_ids, batch_input_mask, batch_segment_ids, batch_p_mask, q_len, q_no

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    @staticmethod
    def collate_fn(data):
        tokens, input_ids, input_mask, segment_ids, p_mask, q_len, q_no = list(zip(*data))

        tokens = list(chain(*tokens))
        input_ids = list(chain(*input_ids))
        input_mask = list(chain(*input_mask))
        segment_ids = list(chain(*segment_ids))
        p_mask = list(chain(*p_mask))
        max_que_length = max(q_len)
        q_mask = []
        q_no_ = []
        for i, q in enumerate(q_len):
            q_mask.append([1] * q + [0] * (max_que_length - q))
            q_no_.append(q_no[i] + [-1] * (max_que_length - q))

        return tokens, torch.tensor(input_ids, dtype=torch.long), torch.tensor(input_mask, dtype=torch.long),\
               torch.tensor(segment_ids, dtype=torch.long), torch.tensor(p_mask, dtype=torch.long), \
               q_len, torch.tensor(q_no_, dtype=torch.long), torch.tensor(q_mask, dtype=torch.long)


# For debugging
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    file_name = "./QuAC_data/dev_10.json"
    dataset = QuestionDataset(file_name, tokenizer, 64, 15)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=QuestionDataset.collate_fn, drop_last=True)

    for i, j in enumerate(loader):
        print(i)
