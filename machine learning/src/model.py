import os

from pytorch_transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer
from pytorch_transformers.modeling_bert import BertForSequenceClassification
from pytorch_transformers.modeling_roberta import RobertaForSequenceClassification
from pytorch_transformers.modeling_distilbert import DistilBertForSequenceClassification


def save_model(model, experiment_name, model_output_dir, epoch=None, tokenizer=None):
    if epoch:
        output_sub_dir = os.path.join(model_output_dir, experiment_name, "epoch_{}".format(epoch))
    else:
        output_sub_dir = os.path.join(model_output_dir, experiment_name)

    os.makedirs(output_sub_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    model_to_save.save_pretrained(output_sub_dir)

    if tokenizer:
        tokenizer.save_pretrained(output_sub_dir)

    return output_sub_dir


def load_model(model_dir, do_lower_case):
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=do_lower_case)

    return model, tokenizer
    
def load_model_roberta(model_dir, do_lower_case):
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    tokenizer = RobertaTokenizer.from_pretrained(model_dir, do_lower_case=do_lower_case)

    return model, tokenizer
    
def load_model_distilbert(model_dir, do_lower_case):
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir, do_lower_case=do_lower_case)

    return model, tokenizer
