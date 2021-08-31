# Predicting the remaining capacity of lithium ion batteries using machine learning

## Summary

This project aims to develop a traditional machine learning model to predict the current state of health (Soh) of a lithium ion battery, using voltage and temperature profiles from discharging cycles. In particular, we aim to predict the battery's remaining capacity in Ah given data from any cycle.

We use pubically available battery data published by NASA at https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#battery. The experimental data consists of groups of experiments performed on Li-ion batteries with a rated capacity of 2 Ah. In particular, batteries 5, 6, 7, and 18 were repeatedly charged and discharged at an ambient temperature of 24C, with a constant discharge current of 2A. The experiments on the remainder of the batteries used in the project were carried out at a temperature of 4C, using the same discharge current.

## Raw data format
