# StageNet: Stage-Aware Neural Networks for Health Risk Prediction

![StageNet Text Logo](images/stagenet.png "StageNet logo")


This repository is an implementation of [StageNet: Stage-Aware Neural Networks for Health Risk Prediction](https://arxiv.org/abs/2001.10054). This project was created as part of UIUC CS 598 Deep Learning for Healthcare Final Project.

## Data Preperation
We're going to be using the MIMC-III dataset for this project. In order for it to be utilized for StageNet though we need to build the benchmark dataset and the decompensation dataset. 
<br><br>
As part of this git repository we've included the [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks/) codebase and made necessary modifications for it to run.
<br><br>
We cannot include the actual dataset as part of this repository due to license restrictions of the MIMIC-III dataset. As a result preparing this data will take several hours depenindong on your compute power.

1. Get access to the MIMIC-III dataset from [here](https://mimic.mit.edu/) and download it locally.
2. Install Python 3.9.13.
3. Navigate to the `mimic3-benchmarks` directory.
4. Run `pip install -r requirements.txt` to install the project dependencies.
5. Extract the MIMIC-III dataset into the folder of your choosing. You will need to reference this path in future steps.
6. Extract the subjets by running:
```bash
python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/
```
7. Fix some of the events with issues and validate they include all the data we need:
```bash
python -m mimic3benchmark.scripts.validate_events data/root/
```
8. Extract episodes from each patient:
```bash
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
```
9. Split the data into a training and test split:
```bash
python -m mimic3benchmark.scripts.split_train_and_test data/root/
```
10. Finally generate the decompensation dataset:
```bash
python -m mimic3benchmark.scripts.create_decompensation data/root/ data/decompensation/
``` 

## Model Requirements
Python 3.7.13 is required to run StageNet. In addition to the required packages below.

To install requirements:
```bash
cd StageNet
pip install -r requirements.txt
```

By default StageNet is configured to use CUDA so ensure that you have the appropriate CUDA driver installed.

## Training

To train the model in the paper using the saved weights, run this command:

```bash
python train.py \
    -test_mode=1 \
    --data_path='./data/' 
```

To train the full model run the following.

```bash
python train.py \
    --data_path='./data/' \
    --file_name='trained_model' 
```

All of the hyperparameters and settings are here:
```
usage: train.py [-h] [--test_mode TEST_MODE] [--data_path <data_path>]
                [--file_name <data_path>]
                [--small_part SMALL_PART]
                [--batch_size BATCH_SIZE]
                [--epochs EPOCHS]
                [--lr LR]
                [--input_dim INPUT_DIM]
                [--rnn_dim RNN_DIM]
                [--output_dim OUTPUT_DIM] [--dropout_rate DROPOUT_RATE]
                [--dropconnect_rate DROPCONNECT_RATE]
                [--dropres_rate DROPRES_RATE]
                [--K K]
                [--chunk_level CHUNK_LEVEL]
```

## Evaluation

When the training commands finish it will output the performance metrics for StageNet. For example, a completed training will output the following:
```
accuracy = 0.9766005873680115
precision class 0 = 0.9877837896347046
precision class 1 = 0.378338098526001
recall class 0 = 0.9883724451065063
recall class 1 = 0.366655558347702
AUC of ROC = 0.9027237860494428
AUC of PRC = 0.3374715298096852
min(+P, Se) = 0.3722376457523598
```

## Pre-trained Models
The pre-trained model/weights can be found in the `saved_weights` folder. The weights file is called `StageNet` and is a pickle model file.

## Results

Our model achieves the following performance:

| Model name         | FI  | Accuracy | AUC | min(+P, Se) |
| ------------------ |---------------- | -------------- | ----- | ------ |
| StageNet   |     0.988         |      0.976       |0.902|0.372


## License
This code base is released under the MIT license as outlined in the `LICENSE` file.

## Contributing
We welcome contributions from other developers! Here are a few ways you can help improve this project:

### Reporting Issues
If you encounter any bugs or problems while using our software, please let us know by opening an issue on GitHub. Be sure to include as much detail as possible, such as the steps to reproduce the problem and any error messages you see.

### Fixing Bugs
If you're able to fix an issue yourself, we encourage you to submit a pull request with your changes.

### Adding Features
If you have an idea for a new feature or improvement, please discuss it with us first by opening an issue. We'd love to hear your ideas and work with you to implement them.

### Improving Documentation
Good documentation is essential for any project, and we welcome contributions to our documentation to help make it more clear and helpful for our users.

Thank you for considering contributing to our project!

## References
>Junyi Gao, Cao Xiao, Yasha Wang, Wen Tang,
Lucas M. Glass, Jimeng Sun. 2020. StageNet:
Stage-Aware Neural Networks for Health Risk Pre-
diction. In Proceedings of The Web Conference
2020 (WWW ’20), April 20–24, 2020, Taipei,
Taiwan. ACM, New York, NY, USA, 11 pages.
https://doi.org/10.1145/3366423.3380136
