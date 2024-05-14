# KAN it be used for Time Series Classification ?

<p align="center">
        <img src="https://api.star-history.com/svg?repos=MSD-IRIMAS/Simple-KAN-4-Time-Series&type=Date" alt="Star History Chart" width="60%"/>
</p>


author : [Ali Ismail-Fawaz](https://hadifawaz1999.github.io/) - [@hadifawaz1999](https://github.com/hadifawaz1999)

In this repository, we present a feature-based Time Series Classifier based on [Kolmogorov–Arnold Networks (KANs)](https://github.com/KindXiaoming/pykan) proposed by [Liu et al. 2024 [1]](https://arxiv.org/pdf/2404.19756).
Given that a KAN model is still a fully connected network, it will not be able to detect temporal dependencies on time series data, however it will be able to work on extracted features from these series.
In this repository, we present a classifier based on extracting the Catch22 features [Lubba et al. 2019 [2]](https://link.springer.com/article/10.1007/s10618-019-00647-x) followed by KAN classifier.
In order to showcase the performance of KAN, we compare it to using a softmax classifier on top of the Catch22 features.
The model of course can work with any type of transformation on time series data, we chose Catch22 for simpliciy.
We use the [aeon](https://github.com/aeon-toolkit/aeon) python package for the feature extraction part.

## Datasets

In this repository we utilize the amazing publicly available [UCR archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/) by [Dau et al. 2019 [3]](https://ieeexplore.ieee.org/abstract/document/8894743/), the largest repository of unvariate Time Series Classification datasets.
Make sure to download the datasets and extract the zip folder.

## Docker image

If you are familiar with docker, simply build the image using the dockerfile provided as follows:

```PS:``` Make sure to adjust the `USER_ID` and `GROUP_ID` arguments in the dockerfile to align with your local machine before building the image.

```docker build -t kan-ts-image .```

Following the image being built, create your docker container and mount the UCR archive datasets into the `/home/myuser/ucr_archive` directory and your current working directory to `/home/myuser/code` directory, as follows:

```docker run --gpus all -it --name kan-ts-container -v "$(pwd):/home/myuser/code" -v "/path/to/ucr/archiev/on/your/machine/:/home/myuser/ucr_archive/" --user $(id -u):$(id -g) kan-ts-image bash```

Make sure to replace `/path/to/ucr/archiev/on/your/machine/` by the directory on your local machine.

Once the containter is created, exit and execute it again in root mode with:

```docker start kan-ts-container```

and

```docker exec -it -u root kan-ts-container bash```

Once in root mode, install gcc and pycatch22 with:

```
$ apg-get install gcc
$ pip install pycatch22
```

## Requirements

```
torch==2.2.1
numpy==1.24.4
pandas==2.0.3
scikit-learn==1.1.3
matplotlib==3.6.2
aeon==0.8.1
hydra-core --upgrade
omegaconf
black==23.11.0
pykan
setuptools==65.5.0
sympy==1.11.1
tqdm==4.66.2
pycatch22
```

## Code Usage

This code uses hydra for the parameters configuration, so simply edit the `config/config_hydra.yaml` file and run the following command:

```python3 main.py```

## Results

The code will generate per dataset a directory containing a csv file, with four columns, accuracy mean and std over five runs on both Catch22+KAN classifier and Catch22+softmax classifier.

Examples of some UCR datasets where KAN improves the classification performance:

| Dataset              | Test Accuracy with KAN | Test Accuracy with softmax  |
| :------------------- | :--------------------: | :-------------------------: |
| Chinatown            |         85.13 %        |           62.09 %           |
| ItalyPowerDemand     |         91.29 %        |           82.41 %           |
| ECG200               |         78.60 %        |           75.00 %           |
| ArrowHead            |         67.99 %        |           63.42 %           |
| CricketX             |         51.90 %        |           35.64 %           |
| CricketY             |         49.13 %        |           32.31 %           |
| CricketZ             |         52.62 %        |           35.90 %           |
| Beef                 |         44.00 %        |           36.67 %           |

## Citing this work

If you use this work, please make sure to cite this code repository as follows:

```
@misc{Ismail-Fawaz2023kan-c22-4-tsc
    author = {Ismail-Fawaz, Ali and Devanne, Maxime and Berretti, Stefano and Weber, Jonathan and Forestier, Germain},
    title = {Feature-Based Time Series Classification with Kolmogorov–Arnold Networks},
    year = {2024},
    publisher = {Github},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/MSD-IRIMAS/Simple-KAN-4-Time-Series}}
}
```

## References

[1] Liu, Ziming, et al. "KAN: Kolmogorov-Arnold Networks." arXiv preprint arXiv:2404.19756 (2024).<br>
[2] Lubba, Carl H., et al. "catch22: CAnonical Time-series CHaracteristics: Selected through highly comparative time-series analysis." Data Mining and Knowledge Discovery 33.6 (2019): 1821-1852.<br>
[3] Dau, Hoang Anh, et al. "The UCR time series archive." IEEE/CAA Journal of Automatica Sinica 6.6 (2019): 1293-1305.

## Acknowledgments

We would like to thank the authors of the Kolmogorov–Arnold Networks paper for their amazing work as well as the authors of the UCR archive for making the Time Series Classification datasets publicly available.
We would also like to thank the Aeon and Pycatch22 python packages for their implementation of the Catch22 feature extractor.
