# Bank Marketing data

`raw/bank-full.csv` is the full 45,211-row dataset from the [UCI Machine Learning Repository Bank Marketing page](https://archive.ics.uci.edu/dataset/222/bank).

Citation: Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. DOI: [10.24432/C5K306](https://doi.org/10.24432/C5K306).

UCI identifies this dataset as CC BY 4.0. Retain this attribution when redistributing it. The benchmark excludes `duration` by default because the duration of the last call is known only after that call and can leak post-contact information into a prediction intended to occur earlier.

Categorical columns are one-hot encoded using categories learned from the training partition only. OSSI `F` groups those expanded columns back to the original input variable, so a categorical variable remains one feature workload regardless of its category count.
