import logging
import os

from pytorch_transformers import BertTokenizer

from config import read_arguments_train, read_arguments_prediction
from data_representation import DeepMatcherProcessor, QqpProcessor
from logging_customized import setup_logging
from src.data_loader import load_data, DataType
from src.model import load_model, load_model_roberta, load_model_distilbert
from src.prediction import predict
from torch_initializer import initialize_gpu_seed

from time import time
import json
import resource
import csv
import fcntl
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, confusion_matrix

setup_logging()

def create_experiment_folder(model_output_dir: str, model_type: str, data_dir: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = "{}__{}__{}".format(data_dir.upper(), model_type.upper(), timestamp)

    output_path = os.path.join(model_output_dir, experiment_name)
    os.makedirs(output_path, exist_ok=True)

    return experiment_name

if __name__ == "__main__":
    args = read_arguments_prediction()
    exp_name = create_experiment_folder(args.model_output_dir, args.model_type, args.data_dir)

    device, n_gpu = initialize_gpu_seed(args.seed)

    if args.model_type == "roberta":
        model, tokenizer = load_model_roberta(os.path.join(args.model_output_dir, args.trained_model_for_prediction), args.do_lower_case)
    elif args.model_type == "distilbert":
        model, tokenizer = load_model_distilbert(os.path.join(args.model_output_dir, args.trained_model_for_prediction), args.do_lower_case)
    else:        
        model, tokenizer = load_model(os.path.join(args.model_output_dir, args.trained_model_for_prediction), args.do_lower_case)
        
    model.to(device)

    if tokenizer:
        logging.info("Loaded pretrained model and tokenizer from {}".format(args.trained_model_for_prediction))
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        logging.info("Loaded pretrained model from {} but no fine-tuned tokenizer found, therefore use the standard tokenizer."
                     .format(args.trained_model_for_prediction))

    if args.data_processor == "QqpProcessor":
        processor = QqpProcessor()
    else:
        # this is the default as it works for all data sets of the deepmatcher project.
        processor = DeepMatcherProcessor()

    test_examples = processor.get_test_examples(args.data_path)

    logging.info("loaded {} test examples".format(len(test_examples)))
    test_data_loader = load_data(test_examples,
                                 processor.get_labels(),
                                 tokenizer,
                                 args.max_seq_length,
                                 args.test_batch_size,
                                 DataType.TEST,
                                 args.model_type)
                                 
                                 
    include_token_type_ids = False
    if args.model_type == 'bert':
       include_token_type_ids = True                             
                                 
    t1 = time()
    simple_accuracy, f1, recall, classification_report, prfs, predictions = predict(model, device, test_data_loader, include_token_type_ids)
    t2 = time()
    testing_time = t2-t1
    test_max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    logging.info("Prediction done for {} examples.F1: {}, Simple Accuracy: {}".format(len(test_data_loader), f1, simple_accuracy))

    logging.info(classification_report)

    logging.info(predictions)

    # save prediction results to file    
    import sys
    import numpy
    import pandas as pd
    numpy.set_printoptions(threshold=sys.maxsize)
    pd.set_option('display.max_colwidth', None)
    
    import os
    from openpyxl import Workbook
    
    prediction_path = os.path.join("predictions", exp_name, "predictions.xlsx")
    
    if not os.path.isfile(prediction_path):
        # create a new Excel workbook
        wb = Workbook()
        # save the workbook to the filename
        wb.save(prediction_path)

    with pd.ExcelWriter(prediction_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        predictions.to_excel(writer, sheet_name=args.model_type)

    
    keys = ['precision', 'recall', 'fbeta_score', 'support']
    prfs = {f'class_{no}': {key: float(prfs[nok][no]) for nok, key in enumerate(keys)} for no in range(2)}
    
    test_score_path = os.path.join("predictions", exp_name, "test_scores.txt")
    with open(test_score_path, 'a') as fout:
        scores = {'simple_accuracy': simple_accuracy, 'f1': f1, 'model_type': args.model_type,
         'data_dir': args.data_dir, 'testing_time': testing_time, 'prfs': prfs}
        fout.write(json.dumps(scores)+"\n\n")
        
    # Generate stats
    predicted_class = predictions['predictions']
    labels = predictions['labels']
    p = precision_score(y_true=labels, y_pred=predicted_class)
    r = recall_score(y_true=labels, y_pred=predicted_class)
    f_star = 0 if (p + r - p * r) == 0 else p * r / (p + r - p * r)
    tn, fp, fn, tp = confusion_matrix(y_true=labels,
                                      y_pred=predicted_class).ravel()
                                      
    result_dict = {
        'method': 'emtransformer-{}-epochs-{}'.format(args.model_type,15),
        'dataset_name': args.data_dir,
        'test_time': round(testing_time, 3),
        'test_max_mem': test_max_mem,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Pre': round(p * 100, 3),
        'Re': round(r * 100, 3),
        'F1':  round(f1 * 100, 3),
        'Fstar': round(f_star * 100, 3)
    }
    logging.info(result_dict)


    # Persist Results
    result_path = os.path.join("predictions", exp_name, "results.csv")
    file_exists = os.path.isfile(result_path)

    with open(result_path, 'a') as results_file:
        heading_list = ['method', 'dataset_name',
                        'test_time',
                        'test_max_mem', 'TP', 'FP',
                        'FN',
                        'TN', 'Pre', 'Re', 'F1', 'Fstar']
        writer = csv.DictWriter(results_file, fieldnames=heading_list)

    if not file_exists:
        writer.writeheader()

    #   fcntl.flock(results_file, fcntl.LOCK_EX)

    writer.writerow(result_dict)

    #   fcntl.flock(results_file, fcntl.LOCK_UN)