# Remaining capacity prediction of Li-ion batteries

## Summary

This project aims to develop a traditional machine learning model using the popular Python library scikit-learn to predict the current state of health (SoH) of a lithium ion battery, using voltage and temperature profiles from discharging cycles. In particular, we aim to predict the battery's remaining capacity in Ah, given data from any cycle.

Using a preliminary random forest model we find that the most important predictive features of remaining capacity are the time taken for the terminal voltage to drop below 3V (75% importance) during discharge, and the time taken for temperature to reach its maximum value during discharge (24% importance). Interestingly, the explicit dependence of ambient temperature of the experiment seems to be negligible. Our final model is a weighted voting ensemble incorporating random forest, extra trees, and XGBoost regressors, achieving a root mean squared error of 0.0160 Ah on the test set.

## Raw data format

We use publically available battery data published by NASA at https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#battery. The experimental data consists of groups of experiments performed on Li-ion batteries with a rated capacity of 2 Ah. In particular, batteries 5, 6, 7, and 18 were repeatedly charged to 4V and discharged at an ambient temperature of 24C, with a constant discharge current of 2A. The experiments on the remainder of the batteries used in the project were carried out at a temperature of 4C, using the same discharge current. The remaining capacities at each cycle were also recorded in addition to the voltage and temperature profiles.

## Results
