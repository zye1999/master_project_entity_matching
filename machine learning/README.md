## Entity Matching with Transformer Architectures

The ```run_all.sh``` file runs the Bert, Roberta and Distilbert models on the dataset in the data folder. To find the relation between the training time and the size of the dataset, we also expand the ```dblp_acm``` dataset by 2, 4, 8, 16 times and run the 3 models on them.
You will find the results in "Experiments" folder. A detailed result report including method, dataset name, train time, memory usage, confusion matrix and f1 score can be find in the ```results.csv``` file in "Experiments" folder.
The code is developed based on paper [Entity Matching with Transformer Architectures - A Step Forward in Data Integration](https://openproceedings.org/2020/conf/edbt/paper_205.pdf). Many thanks to the authors who provide the framework for machine learning code.
