# Remaining capacity prediction of Li-ion batteries

## Summary

This project aims to develop a traditional machine learning model using the popular Python library scikit-learn to predict the current state of health (SoH) of a lithium ion battery, using voltage and temperature profiles from discharging cycles. In particular, we aim to predict the battery's remaining capacity in Ah, given data from any cycle.

Using a preliminary random forest model we find that the most important predictive features of remaining capacity are the time taken for the terminal voltage to drop below 3V (75% importance) during discharge, and the time taken for temperature to reach its maximum value during discharge (24% importance). Interestingly, the explicit dependence of ambient temperature of the experiment seems to be negligible. Our final model is a weighted voting ensemble incorporating random forest, extra trees, and XGBoost regressors, achieving a root mean squared error of 0.0160 Ah on the test set.

## Raw data format and feature extraction

We use publically available battery data published by NASA at https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#battery. The experimental data consists of groups of experiments performed on Li-ion batteries with a rated capacity of 2 Ah. In particular, batteries 5, 6, 7, and 18 were repeatedly charged to 4V and discharged at an ambient temperature of 24C, with a constant discharge current of 2A. The experiments on the remainder of the batteries used in the project were carried out at a temperature of 4C, using the same discharge current. The remaining capacities at each cycle were also recorded in addition to the voltage and temperature profiles. We plot the voltage and temperature discharge profiles of battery #5 as an example below, for various cycle numbers:

![Figure 1](Image/voltage_B0005.png?raw=true "Figure 1: Discharging voltage profile of a typical battery at various cycle numbers")
![Figure 2](Image/temp_B0005.png?raw=true "Figure 2: Discharging temperature profile of a typical battery at various cycle numbers")

## Results

As our test set was stratified to include test examples of cycles from every battery, we can plot the actual measured capacities for each battery against our final model's predicted values for the chosen test examples. The results for a few selected batteries, and the associated errors, are shown below:

![Figure 3](Image/result_B0005.png?raw=true "Figure 3: Battery #5 predicted vs. actual capacities")
![Figure 4](Image/result_B0007.png?raw=true "Figure 4: Battery #7 predicted vs. actual capacities")
![Figure 5](Image/result_B0055.png?raw=true "Figure 5: Battery #55 predicted vs. actual capacities")
![Figure 6](Image/result_B0056.png?raw=true "Figure 6: Battery #56 predicted vs. actual capacities")

## Try it out!

Running the file 'make_prediction.py' will randomly select a battery and cycle number, plot the associated voltage and temperature curves, and use our trained model to predict the capacity.
