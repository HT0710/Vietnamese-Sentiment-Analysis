# Sentiment-Analysis
 
## Main folders
- `datasets`: contains data
- `lightning_log`: contains logs during training
- `lightning_modules`: contains important modules
- `models`: contains models used for training

## Main filess
- `dataset_generator.py`: Used to generate dataset from raw data
- `requirements.txt`: Python packages needed for the project. Install:
    ```bash
    pip install -r requirements.txt
    ```
- `config.yaml`: Configuration for training and testing process
    > **Adjust the config** before training or testing

- `main.py`: Used for training
    ```bash
    python train.py --epoch 10 --learning_rate 0.001 --batch 128
    ```
- `test.py`: Used for testing
    ```bash
    python test.py
    # or
    python test.py --prompt "tuyệt vời"
    ```
