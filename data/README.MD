## Data
All the data in this directory is already in the expected schema and ready to use for the experiments. If you add further data, make sure you convert it to the right schema with the `convert_deepmatcher_files.py` script.

These [datasets](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md) are sourced from [Deep Learning for Entity Matching](https://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf).

## Example: 
``` python
python convert_deepmatcher_files.py --data_dir ../../data/dblp_acm
```

