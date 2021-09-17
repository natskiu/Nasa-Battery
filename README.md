# Experiments with the NASA Li-ion battery dataset
*Venus Lee and Jesse Wang*

This is a repository for two mini machine learning projects using publically available battery data published by NASA at https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#battery. Jump to the following sections to learn more:

1. [Predicting remaining capacity of Li-ion batteries](#1-predicting-remaining-capacity-of-li-ion-batteries)

2. [Predicting future discharging curves from past data](#2-predicting-future-discharging-curves-from-past-data)

## 1. Predicting remaining capacity of Li-ion batteries

This mini-project aims to develop a traditional machine learning model using scikit-learn to predict the current state of health (SoH) of a lithium ion battery, using voltage and temperature profiles from discharging cycles. In particular, we aim to predict the battery's remaining capacity in Ah, given data from any cycle. Our final model is a weighted voting ensemble incorporating random forest, extra trees, and XGBoost regressors, achieving a root mean squared error of 0.0160Ah on the test set.

### Try it out!

- Install the Git large file storage extension at https://git-lfs.github.com/ before cloning the repo to get the ```.pkl``` files. 
- Create a new conda environment using ```conda env create -f environment.yml```. The first line of the ```.yml``` file sets the new environment's name (```batteryenv``` by default). Activate the new environment using ```conda activate batteryenv```.
- Run ```python make_prediction.py``` to randomly select a battery and cycle number, plot the associated voltage and temperature curves, and use our trained model to predict the capacity.

### Feature extraction (feature_extraction.ipynb)

The experimental data consists of groups of experiments performed on Li-ion batteries with a rated capacity of 2Ah. In particular, batteries 5, 6, 7, and 18 were repeatedly charged to 4V and discharged at an ambient temperature of 24C, with a constant discharge current of 2A. The experiments on batteries 49, 50, 51, 53, 54, 55, and 56 were carried out at a temperature of 4C, using the same discharge current. The remaining capacities at each cycle were also recorded, in addition to the voltage and temperature profiles. We plot the voltage and temperature discharge profiles of battery #5 as an example below, for various cycle numbers:

![Figure 1](Image/voltage_B0005.png?raw=true "Figure 1: Discharging voltage profile of a typical battery at various cycle numbers")
![Figure 2](Image/temp_B0005.png?raw=true "Figure 2: Discharging temperature profile of a typical battery at various cycle numbers")

We then extracted the following features from each cycle:
1.	Time taken for the discharging temperature to reach its maximum value
2.	Maximum temperature reached during discharge
3.	Average rate of temperature increase during discharge, as measured by (maximum temperature - initial temperature)/time taken
4.	Time for the measured voltage to drop below 3V
5.	Initial slope of the measured voltage

Initially we considered extracting features from the charging cycles as well; however the data was quite irregular and we determined that the effect on the final model could be ignored. A baseline random forest model indicates that the most important features are #4 (75% importance) and #1 (24% importance). Interestingly, the explicit dependence of ambient temperature of the experiment seems to be negligible.

### Model building (model_building.ipynb)

After removing anomalies using isolation forest methods, we split the data (~1000 data points) into training, validation, and test sets in a 60:20:20 ratio. Each set was stratified according to the amount of cycle data available per battery. We then tried baseline models on default hyperparameters (random forest, extra trees, linear regression, elastic net regression, LGBM, XGBoost, SVM, and k-NN) using 5-fold cross validation on the train set and with RMSE as the evaluation metric. Further to this we selected the best three models - random forest, extra trees, and XGBoost - and performed hyperparameter tuning for each, evaluating the tuned models on the validation set to check for overfitting. Finally we combined the three tuned models into a voting ensemble, whose weights were optimized, and evaluated its performance on the test set.

### Results (evaluation.ipynb)

As our test set was stratified to include test examples of cycles from every battery, we can plot the actual measured capacities for each battery against our final model's predicted values for the chosen test examples. The results for a few selected batteries, and the associated errors, are shown below:

![Figure 3](Image/result_B0005.png?raw=true "Figure 3: Battery #5 predicted vs. actual capacities")
![Figure 4](Image/result_B0007.png?raw=true "Figure 4: Battery #7 predicted vs. actual capacities")
![Figure 5](Image/result_B0055.png?raw=true "Figure 5: Battery #55 predicted vs. actual capacities")
![Figure 6](Image/result_B0056.png?raw=true "Figure 6: Battery #56 predicted vs. actual capacities")

The overall RMSE achieved on the test set of 0.0160Ah is comparable to the error on the validation and training sets, which suggests that an appropriate amount of regularization has been applied to prevent overfitting.

## 2. Predicting future discharging curves from past data

In this mini-project, we train a sequence-to-sequence LSTM network in Keras to predict voltage discharging curves for the next 50 cycles, given 10 cycles' worth of data. The raw data consists of the voltage discharging curves from batteries 5, 6, and 7.

### Try it out!

- Create a new conda environment using ```conda env create -f environment.yml```. The first line of the ```.yml``` file sets the new environment's name (```batteryenv``` by default). Activate the new environment using ```conda activate batteryenv```.
- Change the current directory using ```cd discharge_curve_prediction```.
- Run ```python display_results.py -b <battery_number> -s <starting_cycle>``` to use our trained model to make predictions and visualise them. 
  - ```<battery_number>``` is an integer from ```0``` to ```2```, inclusive (```0``` indicates battery B0005, ```1``` indicates battery B0006, and ```2``` indicates battery B0007).
  - ```<starting_cycle>``` is an integer from ```0``` to ```107```, inclusive. This refers to the first cycle from which data will be sampled.
  - For example, running ```python display_results.py -b 1 -s 90``` will produce the following plots (note that only a subset of the green curves are shown):

![Figure 7](Image/weights_1_dim230.png?raw=true "Figure 7: Actual vs. predicted discharging curves")

## Citation

B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics Data Repository (http://ti.arc.nasa.gov/project/prognostic-data-repository), NASA Ames Research Center, Moffett Field, CA
