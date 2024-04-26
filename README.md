This repository contains the codes developed for the work presented in the following paper:

Palerme, C., Lavergne, T., Rusin, J., Melsom, A., Brajard, J., Kvanum, A. F., Macdonald Sørensen, A., Bertino, L., and Müller, M.: Improving short-term sea ice concentration forecasts using deep learning, The Cryosphere, 2024.

The sub-repositories are:

- AMSR2_on_UNet_grid: contains the codes used for interpolating the AMSR2 sea ice concentration observations on the grid used for the deep learning models

- Benchmark_forecasts: contains the codes used for generating the benchmark forecast called "Anomaly persistence forecasts"

- Figures: contains the codes used for generating the figures of the paper

- Ice_charts_on_UNet_grid: contains the codes used for interpolating the Norwegian ice charts on the grid used for the deep learning models

- Models: contains the codes used for developing the deep learning models and to make predictions using these models, as well as to evaluate the performances of these models

- Ice_edge: contains the codes used for assessing the length of the ice edges from AMSR2 sea ice concentration observations

- Predictor_importances: contains the codes used for assessing the importance of the different predictors

- Standardization: contains the codes used for computing the standardization and normalization statistics of the variables used in the deep learning models

- Training_data: contains the codes used for generating the data used for training the deep learning models, as well as for the validation and test periods
