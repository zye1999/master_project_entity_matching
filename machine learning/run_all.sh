#!/bin/bash
export PYTHONPATH=$(pwd)

cecho(){
    RED="\033[0;31m"
    GREEN="\033[0;32m"
    YELLOW="\033[1;33m"
    # ... ADD MORE COLORS
    NC="\033[0m" # No Color

    printf "${!1}${2} ${NC}\n"
}

SEED=22

# BERT
cecho "GREEN" "BERT"
cecho "YELLOW" "Start abt_buy BERT"
python ./src/run_training.py --model_type=bert --model_name_or_path=bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/abt_buy --train_batch_size=16 --eval_batch_size=16 --max_seq_length=265 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start amazon_google BERT"
python ./src/run_training.py --model_type=bert --model_name_or_path=bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/amazon_google --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start amazon_itunes BERT"
python ./src/run_training.py --model_type=bert --model_name_or_path=bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/amazon_itunes --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_amazon_itunes BERT"
python ./src/run_training.py --model_type=bert --model_name_or_path=bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_amazon_itunes --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start walmart_amazon BERT"
python ./src/run_training.py --model_type=bert --model_name_or_path=bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/walmart_amazon --train_batch_size=16 --eval_batch_size=16 --max_seq_length=150 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_walmart_amazon BERT"
python ./src/run_training.py --model_type=bert --model_name_or_path=bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_walmart_amazon --train_batch_size=16 --eval_batch_size=16 --max_seq_length=150 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dblp_acm BERT"
python ./src/run_training.py --model_type=bert --model_name_or_path=bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dblp_acm --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_dblp_acm BERT"
python ./src/run_training.py --model_type=bert --model_name_or_path=bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_dblp_acm --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dblp_scholar BERT"
python ./src/run_training.py --model_type=bert --model_name_or_path=bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dblp_scholar --train_batch_size=16 --eval_batch_size=16 --max_seq_length=128 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_dblp_scholar BERT"
python ./src/run_training.py --model_type=bert --model_name_or_path=bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_dblp_scholar --train_batch_size=16 --eval_batch_size=16 --max_seq_length=128 --num_epochs=15.0 --seed=${SEED}


# roBERTa
cecho "GREEN" "roBERTa"
cecho "YELLOW" "Start abt_buy roBERTa"
python ./src/run_training.py --model_type=roberta --model_name_or_path=roberta-base --data_processor=DeepMatcherProcessor --data_dir=../data/../data/abt_buy --train_batch_size=8 --eval_batch_size=16 --max_seq_length=265 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start amazon_google roBERTa"
python ./src/run_training.py --model_type=roberta --model_name_or_path=roberta-base --data_processor=DeepMatcherProcessor --data_dir=../data/../data/amazon_google --train_batch_size=8 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start amazon_itunes roBERTa"
python ./src/run_training.py --model_type=roberta --model_name_or_path=roberta-base --data_processor=DeepMatcherProcessor --data_dir=../data/../data/amazon_itunes --train_batch_size=8 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_amazon_itunes roBERTa"
python ./src/run_training.py --model_type=roberta --model_name_or_path=roberta-base --data_processor=DeepMatcherProcessor --data_dir=../data/../data/dirty_amazon_itunes --train_batch_size=8 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start walmart_amazon roBERTa"
python ./src/run_training.py --model_type=roberta --model_name_or_path=roberta-base --data_processor=DeepMatcherProcessor --data_dir=../data/walmart_amazon --train_batch_size=8 --eval_batch_size=16 --max_seq_length=150 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_walmart_amazon roBERTa"
python ./src/run_training.py --model_type=roberta --model_name_or_path=roberta-base --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_walmart_amazon --train_batch_size=8 --eval_batch_size=16 --max_seq_length=150 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dblp_acm roBERTa"
python ./src/run_training.py --model_type=roberta --model_name_or_path=roberta-base --data_processor=DeepMatcherProcessor --data_dir=../data/dblp_acm --train_batch_size=8 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_dblp_acm roBERTa"
python ./src/run_training.py --model_type=roberta --model_name_or_path=roberta-base --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_dblp_acm --train_batch_size=8 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dblp_scholar roBERTa"
python ./src/run_training.py --model_type=roberta --model_name_or_path=roberta-base --data_processor=DeepMatcherProcessor --data_dir=../data/dblp_scholar --train_batch_size=8 --eval_batch_size=16 --max_seq_length=128 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_dblp_scholar roBERTa"
python ./src/run_training.py --model_type=roberta --model_name_or_path=roberta-base --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_dblp_scholar --train_batch_size=8 --eval_batch_size=16 --max_seq_length=128 --num_epochs=15.0 --seed=${SEED}



# DistilBERT
cecho "GREEN" "DistilBERT"
cecho "YELLOW" "Start abt_buy DistilBERT"
python ./src/run_training.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/abt_buy --train_batch_size=16 --eval_batch_size=16 --max_seq_length=265 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start amazon_google DistilBERT"
python ./src/run_training.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/amazon_google --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start amazon_itunes DistilBERT"
python ./src/run_training.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/amazon_itunes --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_amazon_itunes DistilBERT"
python ./src/run_training.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_amazon_itunes --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start walmart_amazon DistilBERT"
python ./src/run_training.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/walmart_amazon --train_batch_size=16 --eval_batch_size=16 --max_seq_length=150 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_walmart_amazon DistilBERT"
python ./src/run_training.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_walmart_amazon --train_batch_size=16 --eval_batch_size=16 --max_seq_length=150 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dblp_acm DistilBERT"
python ./src/run_training.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dblp_acm --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_dblp_acm DistilBERT"
python ./src/run_training.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_dblp_acm --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dblp_scholar DistilBERT"
python ./src/run_training.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dblp_scholar --train_batch_size=16 --eval_batch_size=16 --max_seq_length=128 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_dblp_scholar DistilBERT"
python ./src/run_training.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_dblp_scholar --train_batch_size=16 --eval_batch_size=16 --max_seq_length=128 --num_epochs=15.0 --seed=${SEED}

# Compare the relation between the training time and the size of the dataset
# On the expanded dataset DBLP_ACM_CLEAN
cecho "GREEN" "Expanded dblp_acm"
# BERT
cecho "YELLOW" "Start dirty_dblp_acm BERT"
python ./src/run_training.py --model_type=bert --model_name_or_path=bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_dblp_acm --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_dblp_acm_x2 BERT"
python ./src/run_training.py --model_type=bert --model_name_or_path=bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_dblp_acm_x2 --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_dblp_acm_x4 BERT"
python ./src/run_training.py --model_type=bert --model_name_or_path=bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_dblp_acm_x4 --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_dblp_acm_x8 BERT"
python ./src/run_training.py --model_type=bert --model_name_or_path=bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_dblp_acm_x8 --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_dblp_acm_x16 BERT"
python ./src/run_training.py --model_type=bert --model_name_or_path=bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_dblp_acm_x16 --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

# roBERTa
cecho "YELLOW" "Start dblp_scholar roBERTa"
python ./src/run_training.py --model_type=roberta --model_name_or_path=roberta-base --data_processor=DeepMatcherProcessor --data_dir=../data/dblp_scholar --train_batch_size=8 --eval_batch_size=16 --max_seq_length=128 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dblp_scholar_x2 roBERTa"
python ./src/run_training.py --model_type=roberta --model_name_or_path=roberta-base --data_processor=DeepMatcherProcessor --data_dir=../data/dblp_scholar_x2 --train_batch_size=8 --eval_batch_size=16 --max_seq_length=128 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dblp_scholar_x4 roBERTa"
python ./src/run_training.py --model_type=roberta --model_name_or_path=roberta-base --data_processor=DeepMatcherProcessor --data_dir=../data/dblp_scholar_x4 --train_batch_size=8 --eval_batch_size=16 --max_seq_length=128 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dblp_scholar_x8 roBERTa"
python ./src/run_training.py --model_type=roberta --model_name_or_path=roberta-base --data_processor=DeepMatcherProcessor --data_dir=../data/dblp_scholar_x8 --train_batch_size=8 --eval_batch_size=16 --max_seq_length=128 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dblp_scholar_x16 roBERTa"
python ./src/run_training.py --model_type=roberta --model_name_or_path=roberta-base --data_processor=DeepMatcherProcessor --data_dir=../data/dblp_scholar_x16 --train_batch_size=8 --eval_batch_size=16 --max_seq_length=128 --num_epochs=15.0 --seed=${SEED}

# DistilBERT
cecho "YELLOW" "Start dirty_dblp_acm DistilBERT"
python ./src/run_training.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_dblp_acm --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_dblp_acm_x2 DistilBERT"
python ./src/run_training.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_dblp_acm_x2 --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_dblp_acm_x4 DistilBERT"
python ./src/run_training.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_dblp_acm_x4 --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_dblp_acm_x8 DistilBERT"
python ./src/run_training.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_dblp_acm_x8 --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}

cecho "YELLOW" "Start dirty_dblp_acm_x16 DistilBERT"
python ./src/run_training.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=../data/dirty_dblp_acm_x16 --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0 --seed=${SEED}
