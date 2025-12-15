# Research-project
Team: Oulaya Benani-Dakhama, Oscar Caudreillez, Ulysse Macé, Daniil Notkin, Yuhan Su

## Goals
This project is about a data pipeline that calculates agricultural assets' carbon risk scores, and thus their likelihoods of being stranded.


For now, the main file is in .ipynb format. But in the future, we will change it to a series of .py files, with the main.py file being the main access point.

Our main dataset is ![Agriculture Financial Risk Dataset](https://www.kaggle.com/datasets/programmer3/agriculture-financial-risk-dataset). This dataset contains:
+ Financial and economic data at enterprise level(i.e. revenue, debt-to-equity ratio)
+ Climate & environmental proxies(average temperature, rainfall, etc.)
+ Structural descriptors(i.e. enterprise size)

## Core Assumptions
Given the limits of the available data, we clarify the following:
+ The dataset is enterprise-level, not investor-level
+ Emissions are proxied, not directly observed
+ Carbon prices are scenario-based, not forecasted
+ Our goal is a climate-financial risk pipeline, not precise emissions accounting