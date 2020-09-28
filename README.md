# PyDominion
Python implementation of Dominion for Princeton COS independent work.

1. Create a conda environment with the right dependencies:
```conda env create --name <envname> --file=environment.yml```

2. Edit `aiconfig.py` with paths to store models and data. 

3. Train a basic Monte Carlo agent with default settings: 
```python src/ai.py --save_model --save_data```
