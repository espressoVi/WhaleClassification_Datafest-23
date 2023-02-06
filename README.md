# Team UG&Sons submission for DataFest2023

Implementing a 1D ResNet like architecture for whale sound classification.
Validation accuracy - 99%

## DONE:
* Pre-processing.
* Data loading, etc
* Writing model, training loop, etc
* Test model.
* Write submission code.
* Finetune model.

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
* main.py

