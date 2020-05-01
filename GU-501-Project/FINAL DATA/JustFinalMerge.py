# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:15:59 2019

@author: VChan
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:02:34 2019

@author: VChan
"""


import pandas as pd 
import numpy as np 
import csv

import statistics

import matplotlib.pyplot as plt


# Vid's drug data
filepath1 = "C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/FirstDrugSet2.csv"

yearlydrug = pd.read_csv(filepath1)

# Nicole's pop sex race data
filepath2 = "C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/PopSexRaceEth/PopSexRaceEth-V2.csv"

popsex = pd.read_csv(filepath2)

# Vid's crime data
filepath3 = "C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/UNIFORM CRIME DATA/fullarrestdatajustfips.csv"
fbiarrest = pd.read_csv(filepath3)

# Vid's fips data
filepath4 = "C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/FIPS DATA/statenames.csv"
statenames = pd.read_csv(filepath4)

# Doug's obesity data
filepath5 = "C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/Obesity/CleanObesity.csv"
obesity = pd.read_csv(filepath5)

# Nicole's Politics data
filepath6 = "C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/Politics/politics.csv"
politics = pd.read_csv(filepath6)

# Nicole's poverty and income
filepath7 = "C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/Poverty/poverty_income_county.csv"
povertyincome = pd.read_csv(filepath7)

# Nicole's unemployment
filepath8 = "C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/Unemployment/Unemployment_Yearly.csv"
unemployment = pd.read_csv(filepath8)

# Nicole's Age
filepath9 = "C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/Age/Age.csv"
age = pd.read_csv(filepath9)

# Nicole's Education
filepath10 = "C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/Education/Education_county.csv"
education = pd.read_csv(filepath10)

# Doug's Weather
filepath11 = "C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/Weather/CountyClimateDataClean.csv"
weather = pd.read_csv(filepath11)

# LET'S START MERGING THINGS: 

# Age and Unemploymnet: "unemployment" & "age"

age_unemployment = pd.merge(age, unemployment, left_on=['YEAR','STATE','COUNTY'],right_on=['Year','State','County'],how='left')
age_unemployment.dtypes

# merge age_unemployment with povertyincome

age_unemployment_povertyincome = pd.merge(age_unemployment, povertyincome, left_on =['YEAR','ST_FIPS','CTY_FIPS'] ,right_on = ['Year','ST_FIPS','CTY_FIPS'],how = 'left')
age_unemployment_povertyincome.dtypes

# merge age_unemployment_povertyincome with obesity data

age_unemployment_povertyincome_obesity = pd.merge(age_unemployment_povertyincome,obesity,left_on=['YEAR','ST_FIPS','CTY_FIPS'], right_on=['Year','ST_FIPS','CTY_FIPS'], how='left')
age_unemployment_povertyincome_obesity.dtypes

# merge with education data
age_unemployment_povertyincome_obesity_education = pd.merge(age_unemployment_povertyincome_obesity,education,left_on=['ST_FIPS','CTY_FIPS'], right_on=['ST_FIPS','CTY_FIPS'], how='left')
age_unemployment_povertyincome_obesity_education.dtypes

# merge with weather data
age_unemployment_povertyincome_obesity_education_weather = pd.merge(age_unemployment_povertyincome_obesity_education,weather,left_on=['YEAR','ST_FIPS','CTY_FIPS'], right_on=['Year','State','County'], how='left')
age_unemployment_povertyincome_obesity_education_weather.dtypes

# merge with politics data
age_unemployment_povertyincome_obesity_education_weather_politics = pd.merge(age_unemployment_povertyincome_obesity_education_weather,politics,left_on=['YEAR','ST_FIPS','CTY_FIPS'], right_on=['year','ST_FIPS','CTY_FIPS'], how='left')
age_unemployment_povertyincome_obesity_education_weather_politics.dtypes

#### Finding data types for every data in both files.

yearlydrug.dtypes
popsex.dtypes

fbiarrest.dtypes

statenames.dtypes



#### Find missing values for every data in both files. 

yearlydrug.isnull().sum()
popsex.isnull().sum()

fbiarrest.isnull().sum() 


# GREAT!  No missing values anywhere. 

# Merge the fbi arrest data with state names. 

#fullfbiarrest = pd.merge(fbiarrest,statenames,left_on=['State'],right_on=['State2'], how ='left')

##fbiarrest.isnull().sum()

#fullfbiarrest.to_csv(r'C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/UNIFORM CRIME DATA/fullfbiarrest.csv',index=False)

#fullfbiarrest['CountyName'] = fullfbiarrest['CountyName'].str.upper() 

#### Going to merge the two files together based on county name, state, and year. 

drugpopsex1 = pd.merge(yearlydrug,popsex,left_on=['year','BUYER_STATE','BUYER_COUNTY'],right_on =['YEAR','STATE','COUNTY'], how='left')
drugpopsex1.to_csv(r'C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/drugpopsex1.csv',index=False)

drugpopsex2 = pd.merge(drugpopsex1, fbiarrest, left_on=['year','ST_FIPS','CTY_FIPS'],right_on=['year','FIPS_ST','FIPS_CTY'],how='left')
drugpopsex2.head(10)

finaldrugdata = pd.merge(drugpopsex2,age_unemployment_povertyincome_obesity_education_weather_politics,left_on=['year','ST_FIPS','CTY_FIPS'],right_on=['YEAR','ST_FIPS','CTY_FIPS'],how='left')
#### GETTING DRUG DENSITY

finaldrugdata['drugdensity'] = round(finaldrugdata['DOSAGE_UNIT']/finaldrugdata['TOT_POP'],2)

####### We want to categorize our drug density data into four different bins. 
# Extremely High = 4, High = 3, Medium = 2, Low = 1
# The cut-off is derived from the summary statistic table of the drug density. (Q1, Q2, Q3, and beyond)

conditions = [
        (finaldrugdata['drugdensity']<=2.25),
        (finaldrugdata['drugdensity'] >=2.26) & (finaldrugdata['drugdensity'] <= 3.33),
        (finaldrugdata['drugdensity'] >= 3.34) & (finaldrugdata['drugdensity'] <= 10),
        (finaldrugdata['drugdensity'] >= 10.01) & (finaldrugdata['drugdensity'] <= 1000000)]
choices = [1,2,3,4]
finaldrugdata['densitybins'] = np.select(conditions,choices,default=5)

# turn the densitybins from integer to category instead.

finaldrugdata["densitybins"] = finaldrugdata["densitybins"].astype('category')

####################################################################
# DEALING WITH MISSING VALUES
###################################################################

# Any missing balues in the finaldrugdata data set? 

finaldrugdata.isnull().sum()

# looks like we have some unmatched counties from the population data to the drug data.

# REMOVE SPECIFIC COUNTIES LIKE PR, VI, MP, GU, AND PW.

#removecountylist = ['PR','VI','MP','GU','PW']


#finaldrugdata = finaldrugdata[~finaldrugdata.BUYER_COUNTY.isin(removecountylist)]

finaldrugdata1 = finaldrugdata[(finaldrugdata['densitybins'] != 5)]

#columns = ['Unnamed: 0', 'STATE_x', 'COUNTY_x','YEAR_x','FIPS_ST','FIPS_CTY','Unnamed: 0_x','STATE_y','COUNTY_y','YEAR_y','Unnamed: 0_y','Year_x','State_x','County_x','Unnamed: 0_x','State_y','County_y','Year_y','Unnamed: 0_y','State_x','County_x','Year_x','Unnamed: 0_x','State_y','County_y']
#finaldrugdata1.drop(columns, inplace=True, axis=1)

finaldrugdata1["Median_Income"] = finaldrugdata1["Median_Income"].astype('float64')


finaldrugdata1.to_csv(r'C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/ALMOST FINAL-VID/finaldrugdata1.csv',index=False)

################# JUST DRUG DENSITY DATA ONLY WITH NO MISSING VALUES:
subsetdrugdensityonlycol = finaldrugdata1[['BUYER_COUNTY','BUYER_STATE','year_x','DOSAGE_UNIT','TOT_POP','drugdensity','densitybins',
                                           'TOT_MALE','TOT_FEMALE','WA_PER','BA_PER','IA_PER','H_PER','GRNDTOT','Poverty_Percent',
                                           'Median_Income','ltHS','HS','sCorA','B+','ObesityPercent','democrat','republican','Unemp_Yearly_Rate']]

subsetdrugdensityonlycol.isna().sum()

# Replacing missing values by using  county mean of each variable. 

subsetdrugdensityonlycol["GRNDTOT"].fillna(subsetdrugdensityonlycol.groupby(["BUYER_STATE","BUYER_COUNTY"])["GRNDTOT"].transform("mean"),inplace=True)
subsetdrugdensityonlycol["ObesityPercent"].fillna(subsetdrugdensityonlycol.groupby(['BUYER_STATE',"BUYER_COUNTY"])["ObesityPercent"].transform("mean"),inplace=True)
subsetdrugdensityonlycol["Poverty_Percent"].fillna(subsetdrugdensityonlycol.groupby(['BUYER_STATE',"BUYER_COUNTY"])["Poverty_Percent"].transform("mean"),inplace=True)
subsetdrugdensityonlycol["Median_Income"].fillna(subsetdrugdensityonlycol.groupby(['BUYER_STATE',"BUYER_COUNTY"])["Median_Income"].transform("mean"),inplace=True)

subsetdrugdensityonlycol["democrat"].fillna(subsetdrugdensityonlycol.groupby(['BUYER_STATE',"BUYER_COUNTY"])["democrat"].transform("mean"),inplace=True)
subsetdrugdensityonlycol["republican"].fillna(subsetdrugdensityonlycol.groupby(['BUYER_STATE',"BUYER_COUNTY"])["republican"].transform("mean"),inplace=True)
subsetdrugdensityonlycol["Unemp_Yearly_Rate"].fillna(subsetdrugdensityonlycol.groupby(['BUYER_STATE',"BUYER_COUNTY"])["Unemp_Yearly_Rate"].transform("mean"),inplace=True)

# Check to see the missing values again. Seem like there are still some. 
subsetdrugdensityonlycol.isna().sum()

# I would prefer the state average but democrat, republican, and unemployment data in AK are not totally available. So I had to use the national average for each year.
subsetdrugdensityonlycol["GRNDTOT"].fillna(subsetdrugdensityonlycol.groupby("year_x")["GRNDTOT"].transform("mean"),inplace=True)
subsetdrugdensityonlycol["democrat"].fillna(subsetdrugdensityonlycol.groupby("year_x")["democrat"].transform("mean"),inplace=True)
subsetdrugdensityonlycol["republican"].fillna(subsetdrugdensityonlycol.groupby("year_x")["republican"].transform("mean"),inplace=True)
subsetdrugdensityonlycol["Unemp_Yearly_Rate"].fillna(subsetdrugdensityonlycol.groupby("year_x")["Unemp_Yearly_Rate"].transform("mean"),inplace=True)

# Missing values still exist?
subsetdrugdensityonlycol.isna().sum() # doesn't seem like. Great!


## Instead of dropping all of them, let's try to impute them using the federal average. 
#
#print(subsetdrugdensityonlycol.shape)
#subsetdrugdensityonlycol.dropna(inplace=True)
#print(subsetdrugdensityonlycol.shape)

subsetdrugdensityonlycol['crimerate'] = round(subsetdrugdensityonlycol['GRNDTOT']/subsetdrugdensityonlycol['TOT_POP'],2)

subsetdrugdensityonlycol.to_csv(r'C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/ALMOST FINAL-VID/subsetdrugdensityonlycol.csv',index=False)

averagedrugdensitybyyear = subsetdrugdensityonlycol.groupby(['BUYER_COUNTY','year_x']).mean()

averagedrugdensitybyyear.to_csv(r'C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/ALMOST FINAL-VID/averagedrugdensitybyyear.csv',index=False)

################# Chosen counties to investigate based on the information from my analysis in Tableau:

#top5counties = finaldrugdata1.BUYER_COUNTY[['NORTON CITY','CHARLESTON','MARTINSVILLE CITY','LEAVENSWORTH','MINGO']]

#mid5counties = finaldrugdata1.BUYER_COUNTY[['TISHOMINGO','WALKER','COFFEE','YANCEY','GEORGETOWN']]

#low5counties = finaldrugdata1.BUYER_COUNTY[['LOS ANGELES','RAMSEY','MOODY','YANCEY','WASHINGTON DC']]

### Let's investigate Charleston SC. 
### According to https://geology.com/county-map/south-carolina.shtml, its surrounding counties are: 
### Dorchester, Colleton, Berkeley, Beaufort

### Let's investigate Mingo, WV. 
### According to the Tableau map, its surrounding counties are:
### Logan, Pike, Martin, Wayne, Lincoln, Buchanan, Mcdowell, Wyoming


### Let's investigate Walker, AL:
### According to the Tableau map, its surrounding counties are:
### ['Winston','Cullman','Jefferson','Blount','Marion','Fayette','Tuscaloosa']

### Let's investigate Coffee, TN:
### According to the Tableau map, its surroundingn counties are:
### ['Warren','Grundy','Cannon','Rutherford','Benford','Moore','Franklin']


### Let's investigate Los Angeles, CA:
### According to the Tableau map, its surrounding counties are:
### ['Kern','Ventura','San Bernardino','Orange']

### Let's investigate Ramsey, MN:
### According to the Tableau map, its surrounding counties are: 
### ['Anoka','Washington','Dakota','Hennepin']

### Chosen states and counties to investigate: 

###################################################################################




# WHITE SPACE





##### LET'S EXAMINE DRUG DENSITY
################################################################################
#
## using 'describe' function in pandas.
#finaldrugdata1['TOT_POP'].describe()
#
#finaldrugdata1['DOSAGE_UNIT'].describe()
#
#finaldrugdata1['drugdensity'].describe() # the range of drug density is "wow". There is at least one county with 36.89% density!
#
## let's examine it further by year. 
#
#finaldrugdata1.groupby('year')['drugdensity'].describe()
#
## Find the top 100 counties that have high drug density values regardless of which year. 
## The goal is to find the top 20 counties but to make sure we have 20 unique counties, we need to reach further.
#
#topcounties = finaldrugdata1.nlargest(100,['drugdensity'])
#topcounties.to_csv(r'C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/topcounties.csv',index=False)
#
## Find how many unique counties by states in the "topcounties" dataframe. 
#
#uniquetopcounties = topcounties.groupby(['BUYER_COUNTY','BUYER_STATE']).size().reset_index(name='Freq')
#
#uniquetopcounties.to_csv(r'C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/uniquetopcounties.csv',index=False)

# For the period of our study, two counties from VA appear 7 times in the top 100. 
# Followed by Kentucky.



#################################################################################################################
subsetdrugdensityonlycol['FEMALE_PER'] = round(subsetdrugdensityonlycol['TOT_FEMALE']/subsetdrugdensityonlycol['TOT_POP'],2)
subsetdrugdensityonlycol['TOT_MALE'] = round(subsetdrugdensityonlycol['TOT_MALE']/subsetdrugdensityonlycol['TOT_POP'],2)
subsetdrugdensityonlycol['HSorLess'] = subsetdrugdensityonlycol['ltHS'] + subsetdrugdensityonlycol['HS']
subsetdrugdensityonlycol['CollegeorMore'] =subsetdrugdensityonlycol['sCorA'] + subsetdrugdensityonlycol['B+']
################################################################################################################


#### PRINT OUT THE FINAL DATA FILE: 

subsetdrugdensityonlycol.to_csv(r'C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/ALMOST FINAL-VID/subsetdrugdensityonlycol.csv',index=False)

### Chosen states and counties to investigate: 

chosenstates = ['SC','WV','AL','TN','CA','MN']
chosencounties = ['CHARLESTON','MINGO','WALKER','COFFEE','LOS ANGELES','RAMSEY','Dorchester','Colleton','Berkeley','Beaufort','Logan','Pike','Martin','Wayne','Lincoln','Buchanan','Mcdowell','Wyoming',
                  'Winston','Cullman','Jefferson','Blount','Marion','Fayette','Tuscaloosa','Warren','Grundy','Cannon','Rutherford','Benford','Moore','Franklin',
                  'Kern','Ventura','San Bernardino','Orange','Anoka','Washington','Dakota','Hennepin']
chosencountiesupper = [x.upper() for x in chosencounties]

subsetdrugdensityonlycol.BUYER_STATE.isin(chosenstates)
subsetdrugdensityonlycol_chosenstates= subsetdrugdensityonlycol[subsetdrugdensityonlycol.BUYER_STATE.isin(chosenstates)]

subsetdrugdensityonlycol_chosenstates.to_csv(r'C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/ALMOST FINAL-VID/subsetdrugdensityonlycol_chosenstates.csv',index=False)

subsetdrugdensityonlycol_chosenstatescounties= subsetdrugdensityonlycol_chosenstates[subsetdrugdensityonlycol_chosenstates.BUYER_COUNTY.isin(chosencountiesupper)]

subsetdrugdensityonlycol_chosenstatescounties.to_csv(r'C:/Users/VChan/Desktop/GeorgetownAnalytics/Analytics501/Project/ALMOST FINAL-VID/subsetdrugdensityonlycol_chosenstatescounties.csv',index=False)