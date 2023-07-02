<h1 align="center">Implementation and evaluation of entity matching algorithms</h1>
<h2 align="center"> :pencil: Introduction</h2>

Entity matching is an important task of distinguishing whether text pairs refer to the same entity. In this repository, we implement text-based methods and machine learning methods to solve this problem. Also, their accuracy and runtime are compared.

<h2 align="center"> :pencil: Download and use</h2>
You can use Jupiter Notebook, vscode, or other IDEs to run these codes.

Please enter
```bash
$ git clone http://GitHub.com/zye1999/master-project-entity-matching.git
```
in the terminal to download.

<h2 align="center"> :pencil: Repository structure</h2>

### Datasets
In folder “data”, there are four clean datasets and four corresponding dirty datasets which are analyzed in the paper. These [datasets](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md) are sourced from [Deep Learning for Entity Matching](https://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf).

### Text-based methods
- Folder “crossparsing” contains algorithms of crossparsing method to solve entity matching problems.
- Folder "character based model" contains algorithm of character based model based on Levenshtein distance.
- Folder "token-based” contains three token-based algorithms based on 3 grams.

### Machine learning methods
The folder "machine learning" contains three algorithms (Bert, Roberta, and Distilbert).
To reproduce the experiments, run the `run_all.sh` file.

###  Evaluation and comparison
In the “evaluation and comparison” folder, there is a file(plot_result.ipynb) that compares the accuracy of three text based methods.

The “NEW text-based methods runtime.ipynb” file shows the runtime and f1 score of text-based methods.

The “text-based methods errors.ipynb” file shows all the mismatch cases of text based methods.

<h2 align="center"> :pencil: Contributions</h2>

| Name        | Personal page                                                                                                                                  |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Xiaonan Jian    | [![GitHub Badge](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/xiaonan-jian)   |
| Jingjing Li | [![GitHub Badge](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Jing-jing-Li) |
| Geyu Meng   | [![GitHub Badge](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/GY-Meng)      |
| Zi Ye  | [![GitHub Badge](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/zye1999)     |

<h2 align="center"> :pencil: Acknowledgement</h2>

- Many thanks to Dr. Sven Helmer who helped us a lot as a tutor during the research.
- The part of machine learning code is developed based on paper [Entity Matching with Transformer Architectures - A Step Forward in Data Integration](https://openproceedings.org/2020/conf/edbt/paper_205.pdf). Many thanks to the authors who provide the framework for machine learning code.
