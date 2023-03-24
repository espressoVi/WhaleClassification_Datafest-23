# Team UG&Sons submission for DataFest2023 - Whale Classification challenge
# 1st Runner-up.

Implementing a 1D ResNet like architecture for whale sound classification.
Validation accuracy - 99%
Test F1 - 0.985

## USAGE:
* Put training data in folder ./data/train and test data in folder ./data/test
* The files in all directories are renamed by appending label_ to the filename, i.e.
	```filename.wav --> 0_filename.wav```
	for files with label 0.

* For test files we append the id instead in a similar manner.
	```testFilename.wav --> testFilename_testFilename.wav```

* Dependencies are mentioned in dependencies.yml file. Can be installed with
	```conda env create -f dependencies.yml```
* Training followed by writing predictions can be done via
	```python main.py```

## FILES:
* main.py > calls everything to train and write predictions.
* config.toml > Toml file containing constants and other hyperparameter values.
* train_test.py > Contains training, evaluation, etc routines.
* dataset.py > Loads files and does the preprocessing.
* models/modelBlocks.py > Contains the pieces that make the model.
* models/models.py > Contains the 1D deep CNN model.
* final-model > Contains the weights for our best model.
* data > Directory with data

## DONE:
* Pre-processing.
* Data loading, etc
* Writing model, training loop, etc
* Test model.
* Write submission code.
* Finetune model.
