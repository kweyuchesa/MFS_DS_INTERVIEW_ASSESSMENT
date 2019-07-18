# MFS_DS_INTERVIEW_ASSESSMENT
    MOBILE FINANCIAL SYSTEM SENT A DATA-SET TO TEST MY DATA SCIENCE AND ANALYTICS SKILLS TOWARDS  A DATA SCIENCE ROLE AT THE COMPANY. THIS REPOSITORY OUTLINES THE PROJECT DETAILS AND OUTCOMES OF MY ANALYSIS PLUS AN END TO END REPORT IN THE READ ME . 
    

## Import the libraries 
    import numpy as np
    import pandas as pd
    import pandas_profiling
    import seaborn as sns 
    
## Standardization of the Dataset was done in Excel
    Immediately I received the dataset, the variables: FEATURE 1-FEATURE3 had huge values in billions and hence they needed to be reduced through the =Standardize Fuction in Excel. 
    As a result i got small values to be worked on and ready to be fit on any linear regression model. 
    I saved the new Assignment dataset as "Assignment_std.csv"
    
## Reading the dataset
    df = pd.read_csv("Assignment_std.csv",index_col=None, na_values=['NA'])
    df
    
## Exploring a few rows of the imported dataset
    df.head()
    
## EDA with pandas profiling 
    pandas_profiling.ProfileReport(df)

## exporting the profiling report into html report
    profile = pandas_profiling.ProfileReport(df)
    profile.to_file(outputfile="Assignment_std_mfs.html")

## Checking if there is any missing values 
    df.isnull().any()
    
## converting F2 categorical variable to floats variables - for easier Analysis. 
    cols = ['Output1', 'Output2']
    for col in cols:
        df[col] = df[col].astype(dtype=np.int64)
    df["F2"] = pd.to_numeric(df.F2, errors='coerce')
    #df["Date"] = pd.to_numeric(df.Date, errors='coerce')
    
## Ascerting Conversion of our columns 
    df.info()
    
## Cleaning dataset of NaN values 

    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)
## Exploring Datasset - using group by to calculate average Output1 in the 4 days
    output1_avg = df.groupby(['Date'])['Output1'].mean()
    output1_avg  
    
## using group by to calculate average time/Output in the 4 days
    output2_avg = df.groupby(['Date'])['Output2'].mean()
    output2_avg   

## Preparation of variables for linear regression model
    import statsmodels.formula.api as smf
    X = df[['F1','F2','F3']]
    y = df[['Ouput1']]
    
## Fitting an OLS model with F1 F2 F3 as the intercepts 
    X = sm.add_constant(X)
    est = smf.ols(formula = 'Ouput1 ~ F1 + F2 + F3', data = df).fit()
    est.summary()

## saving our dataset to csv in our working directory
    df.to_csv('mfs.csv')
