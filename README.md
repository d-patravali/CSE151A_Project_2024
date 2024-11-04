# CSE151A_Project_2024
Stock Market Prediction Model for CSE 151A Fall 2024

Link to Data File: https://drive.google.com/file/d/1TNiScpuu1YHd3VzpKw-R64aatgQNukcD/view?usp=sharing

Link To Google Colab For Milestone 2: https://colab.research.google.com/drive/1LGUcWIPsxZ01dDa88lvVxoKfzZO5921M?usp=sharing
(Can only be accessed when using UCSD email)


Milestone 2 Q5: How will you preprocess your data? You should only explain (do not perform pre-processing as that is in MS3) this in your README.md file and link your Jupyter notebook to it. All code and  Jupyter notebooks have be uploaded to your repo.

The first steps of our preprocessing will be to handle the outliers we have. From the graphing in this step, we have a fair amount, so we need to adaquitley handle them for our model. We also have a couple missing values in two of our features, so we will explore leaving them be and adding in average values at the missing spots. For our features that are skewed we want to experiment with normalizing said data when employing it in the model. Additionally, in the pre-processing step we are considering adding in a couple new features that highlight price precentage changes from day to day and throughout each day. We will likely use a 80-10-10 split for training, validation, and testing, but want to try a couple different splits to find an optimal approach. 
