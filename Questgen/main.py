import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import random
import spacy
from sense2vec import Sense2Vec
from collections import OrderedDict
import string
import pke
import nltk
import numpy 
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.corpus import brown
from similarity.normalized_levenshtein import NormalizedLevenshtein
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
from Questgen.encoding.encoding import beam_search_decoding
from Questgen.mcq.mcq import tokenize_sentences
from Questgen.mcq.mcq import get_keywords
from Questgen.mcq.mcq import get_sentences_for_keyword
from Questgen.mcq.mcq import generate_questions_mcq
from Questgen.mcq.mcq import generate_normal_questions
from concurrent.futures import ThreadPoolExecutor
import math

class QGen:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        self.model = T5ForConditionalGeneration.from_pretrained('Parth/result')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.nlp = spacy.load('en_core_web_sm')
        self.s2v = Sense2Vec().from_disk('s2v_old')
        self.fdist = FreqDist(brown.words())
        self.normalized_levenshtein = NormalizedLevenshtein()
        self.set_seed(42)
        
    def set_seed(self, seed):
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    def predict_mcq_parallel(self, payload):
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions", 4)
        }
        text = inp['input_text']
        sentences = tokenize_sentences(text)
        num_parts = math.ceil(math.sqrt(len(sentences)))
        parts = self.split_text_into_parts(sentences, num_parts)
        final_output = {}
        with ThreadPoolExecutor(max_workers=num_parts) as executor:
            results = executor.map(self.process_part_for_mcq, parts)
            questions = []
            for result in results:
                if result:
                    questions.extend(result["questions"])
            final_output["statement"] = text
            final_output["questions"] = questions
        return final_output
    
    def split_text_into_parts(self, sentences, num_parts):
        part_size = math.ceil(len(sentences) / num_parts)
        parts = [sentences[i:i + part_size] for i in range(0, len(sentences), part_size)]
        return parts
    
    def process_part_for_mcq(self, part_sentences):
        joiner = " "
        modified_text = joiner.join(part_sentences)
        keywords = get_keywords(self.nlp, modified_text, inp['max_questions'], self.s2v, self.fdist, self.normalized_levenshtein, len(part_sentences))
        keyword_sentence_mapping = get_sentences_for_keyword(keywords, part_sentences)
        for k in keyword_sentence_mapping.keys():
            text_snippet = " ".join(keyword_sentence_mapping[k][:3])
            keyword_sentence_mapping[k] = text_snippet
        final_output = {}
        if len(keyword_sentence_mapping.keys()) == 0:
            return final_output
        else:
            try:
                generated_questions = generate_questions_mcq(keyword_sentence_mapping, self.device, self.tokenizer, self.model, self.s2v, self.normalized_levenshtein)
            except:
                return final_output
            final_output["questions"] = generated_questions["questions"]
        return final_output

class BoolQGen:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_boolean_questions')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.set_seed(42)
        
    def set_seed(self, seed):
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def random_choice(self):
        a = random.choice([0,1])
        return bool(a)
    
    def predict_boolq_parallel(self, payload):
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions", 4)
        }
        text = inp['input_text']
        num_parts = math.ceil(math.sqrt(len(text)))
        parts = self.split_text_into_parts(text, num_parts)
        final_output = {}
        with ThreadPoolExecutor(max_workers=num_parts) as executor:
            results = executor.map(self.process_part_for_boolq, parts)
            boolean_questions = []
            for result in results:
                if result:
                    boolean_questions.extend(result["Boolean Questions"])
            final_output["Text"] = text
            final_output["Boolean Questions"] = boolean_questions
        return final_output

    def split_text_into_parts(self, text, num_parts):
        part_size = math.ceil(len(text) / num_parts)
        parts = [text[i:i + part_size] for i in range(0, len(text), part_size)]
        return parts
    
    def process_part_for_boolq(self, part_text):
        sentences = tokenize_sentences(part_text)
        joiner = " "
        modified_text = joiner.join(sentences)
        answer = self.random_choice()
        form = "truefalse: %s passage: %s </s>" % (modified_text, answer)
        encoding = self.tokenizer.encode_plus(form, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
        output = beam_search_decoding(input_ids, attention_masks, self.model, self.tokenizer)
        final_output = {"Boolean Questions": output}
        return final_output

class AnswerPredictor:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-large', model_max_length=512)
        self.model = T5ForConditionalGeneration.from_pretrained('Parth/boolean')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.set_seed(42)
        
    def set_seed(self, seed):
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def predict_answer_parallel(self, payload):
        inp = {
            "input_text": payload.get("input_text"),
            "input_question": payload.get("input_question")
        }
        num_parts = math.ceil(math.sqrt(len(inp["input_question"])))
        parts = self.split_questions_into_parts(inp["input_question"], num_parts)
        answers = []
        with ThreadPoolExecutor(max_workers=num_parts) as executor:
            results = executor.map(self.process_part_for_answer, zip(parts, [inp["input_text"]] * len(parts)))
            for result in results:
                answers.extend(result)
        return answers

    def split_questions_into_parts(self, questions, num_parts):
        part_size = math.ceil(len(questions) / num_parts)
        parts = [questions[i:i + part_size] for i in range(0, len(questions), part_size)]
        return parts

    def process_part_for_answer(self, part_data):
        part_questions, text = part_data
        answers = []
        for ques in part_questions:
            context = text
            question = ques
            input_text = "question: %s <s> context: %s </s>" % (question, context)
            encoding = self.tokenizer.encode_plus(input_text, return_tensors="pt")
            input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
            greedy_output = self.model.generate(input_ids=input_ids, attention_mask=attention_masks, max_length=256)
            answer = self.tokenizer.decode(greedy_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            answers.append(answer.strip().capitalize())
        return answers
