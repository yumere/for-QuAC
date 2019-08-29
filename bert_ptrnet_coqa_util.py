from __future__ import absolute_import, division, print_function

import json
import logging
import os
import pickle
from random import shuffle
from typing import List
import random

import numpy as np
import torch
from dataclasses import dataclass
from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


@dataclass
class QuestionSequence:
    q_turn_id: List[int]
    q_text_list: List[str]
    tokens: List[List[str]]
    input_ids: List[List[int]]
    input_mask: List[List[int]]
    segment_ids: List[List[int]]
    p_mask: List[List[int]]
    question_mask: List[int]
    target: List[int]


def prepare_dataset(file_path, tokenizer: BertTokenizer, max_question_len, max_sequence_len, samples_no=5,
                 cls_token="[CLS]", sep_token="[SEP]", pad_token=0, sequence_a_segment_id=0, cls_token_segment_id=0,
                 pad_token_segment_id=0, mask_padding_with_zero=True, save_path=None, in_answer=True):
    # Must exist dataset
    assert os.path.exists(file_path), "{} not exists".format(file_path)
    dataset = json.load(open(file_path))['data']

    def is_whitespace(char):
        if char == " " or char == "\t" or char == "\r" or char == "\n" or ord(char) == 0x202F:
            return True
        return False

    results = []
    # TODO: need the multi-processing by using joblib
    for d_id, d in enumerate(tqdm(dataset, desc="Dataset", ncols=85)):
        questions = [q['input_text'] for q in d['questions']]
        answers = [(a['span_text'], a['input_text']) for a in d['answers']]
        # TODO: Make all possible sequences
        questions = questions[:max_question_len]
        answers = answers[:max_question_len]
        questions_no = list(range(len(questions)))
        questions_len = len(questions)

        q_doc_tokens = []
        q_char_to_word_offset = []

        for question, answer in zip(questions, answers):
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True

            # TODO: check use span_text or input_text
            context = " ".join([question, answer[0]]) if in_answer else question
            for c in context:  # 0 means that I use span_text for the answer text
                if is_whitespace(c):
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
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            # TODO: Need to apply doc stride
            all_doc_tokens = all_doc_tokens[:max_sequence_len - 2]

            tokens = []
            segment_ids = []
            p_mask = []

            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            p_mask.append(0)

            tokens += all_doc_tokens + [sep_token]
            segment_ids += [1] * len(all_doc_tokens) + [0]  # Question tokens are masked as 1
            p_mask += [1] * len(all_doc_tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            while len(input_ids) < max_sequence_len:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_sequence_len
            assert len(input_mask) == max_sequence_len
            assert len(segment_ids) == max_sequence_len

            batch_tokens.append(tokens)
            batch_input_ids.append(input_ids)
            batch_input_mask.append(input_mask)
            batch_segment_ids.append(segment_ids)
            batch_p_mask.append(p_mask)

        for _ in range(samples_no):
            # TODO: Need to make the samples to have a maximum hamming distance
            questions_no = random.sample(questions_no, len(questions_no))
            target = np.array(questions_no).argsort().tolist()
            questions_ = [questions[i-1] for i in questions_no]
            batch_tokens_ = [batch_tokens[i-1] for i in questions_no]
            batch_input_ids_ = [batch_input_ids[i-1] for i in questions_no]
            batch_input_mask_ = [batch_input_mask[i-1] for i in questions_no]
            batch_segment_ids_ = [batch_segment_ids[i-1] for i in questions_no]
            batch_p_mask_ = [batch_p_mask[i-1] for i in questions_no]
            question_mask = [1] * questions_len

            results.append(QuestionSequence(q_turn_id=questions_no, q_text_list=questions_, tokens=batch_tokens_,
                                            input_ids=batch_input_ids_, input_mask=batch_input_mask_,
                                            segment_ids=batch_segment_ids_, p_mask=batch_p_mask_,
                                            question_mask=question_mask, target=target))

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            if os.path.dirname(save_path) != "":
                os.makedirs(os.path.dirname(save_path))

        with open(save_path, "wb") as f:
            f.write(pickle.dumps(results))

    return results


class CoQAOrderDataset(Dataset):
    def __init__(self, json_file=None, pkl_file=None, do_lower_case=True, **kwargs):
        super(CoQAOrderDataset, self).__init__()
        self.dataset: List[QuestionSequence]
        if os.path.exists(pkl_file):
            self.dataset = pickle.load(open(pkl_file, "rb"))
        else:
            tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=do_lower_case)
            self.dataset = prepare_dataset(json_file, tokenizer, save_path=pkl_file, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    @staticmethod
    def collate_fn(data: List[QuestionSequence]):
        input_ids, input_mask, segment_ids, targets = [], [], [], []
        q_turn_ids = []
        question_mask = []

        for d in data:
            input_ids.append(d.input_ids)
            input_mask.append(d.input_mask)
            segment_ids.append(d.segment_ids)
            targets.append(d.target)
            q_turn_ids.append(d.q_turn_id)
            question_mask.append(d.question_mask)

        max_q_len = max(len(i) for i in input_ids)
        for i in range(len(input_ids)):
            cur_len = len(input_ids[i])

            input_ids[i] += [[0]*24] * (max_q_len - cur_len)
            input_mask[i] += [[0]*24] * (max_q_len - cur_len)
            segment_ids[i] += [[0]*24] * (max_q_len - cur_len)
            targets[i] += [-1] * (max_q_len - cur_len)
            question_mask[i] += [0] * (max_q_len - cur_len)

        return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids), \
               torch.tensor(targets), torch.tensor(question_mask)


if __name__ == '__main__':
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    # file_name = "./CoQA-dataset/dev.json"
    # save_path = "./coqa-dev.pkl"
    #
    # results = prepare_dataset(file_name, tokenizer, max_question_len=15, max_sequence_len=24, samples_no=10, save_path=save_path)
    # print("results: {:,}".format(len(results)))

    from torch.utils.data import DataLoader
    dataset = CoQAOrderDataset(json_file="./coqa-dataset/dev.json", pkl_file="./coqa-dev.pkl",
                               max_question_len=20, max_sequence_len=24, samples_no=5)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=CoQAOrderDataset.collate_fn)
    for i, d in enumerate(tqdm(loader)):
        if i == 10:
            break
        print(d)
