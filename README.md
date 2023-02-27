------------
LuckMa Machine Learning Library Showcase Overview - https://www.luckma.io
------------
- Multiclass machine learning models compilation architecture - https://github.com/racesmoky/luckma-ats-docker/blob/main/ml/code/build.py
- Spark Data Bigdata Transformation - https://github.com/racesmoky/luckma-ats-docker/blob/main/ml/code/transform.py
- Tensorflow RNN estimator implementation - https://github.com/racesmoky/luckma-ats-docker/blob/main/ml/code/train.py
- Tensorflow Serving CPU/GPU using Docker - https://github.com/racesmoky/luckma-ats-docker/blob/main/ml/code/serve.py
- Shell script to install all NVidia Cuda libraries - https://github.com/racesmoky/luckma-ats-docker/blob/main/ml/build/install_packages.sh

------------
Project Organization
------------

    ├── ml                             <- Folder for all ML related files
    │
    ├── .env                           <- Python dotenv configuration file, contains environment specific configuration
    ├── .gitignore                     <- Lists the files to ignore on git commit
    ├── LICENSE                        <- License file for the project
    ├── docker-compose.yml             <- Docker compose build configuration
    ├── environment.yml                <- Lists python dependencies for yamel
    ├── model_config.config            <- Contains the list of models compilings, allowing multi model training and compilation
    └── requirements.txt               <- Lists python dependencies for the whole project



--------
