# human-aligned-calibration

## Install and download prerequisites

Install required packages with `pip install -r requirements.txt`

Download Human-AI Interactions Dataset and dependency `haiid.py` (used to load dataset in `hac.py`) from https://github.com/kailas-v/human-ai-interactions into the directory `./human_ai_interactions_data` :

```{r, engine='bash'}
├── human_ai_interactions_data
│   ├── haiid.py
│   ├── haiid_dataset.csv
```

## Repository Structure

- `hac.py`: code to pre-process the dataset and run experiment on "Art", "Sarcasm", "Cities" and "Census" tasks
- `./plots`: contains sub-directories `./barplot`, `./roc` and `./hist` where corresponding plots are saved to

## Running the Experiment 

- Run: 
    ```{r, engine='bash'} 
        python3 hac.py
    ```
- Output:
    - Table of metrics for all tasks printed to console
    - Generated plots saved under `./plots` sub-directories




