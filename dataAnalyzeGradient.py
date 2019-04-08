import pandas as pd
import matplotlib.pyplot as plt



### ANALISE EXPLORATORIA
dfTrain = pd.read_csv('treino_new.csv')

dfStores = pd.read_csv('lojas.csv')

dfNew = pd.merge(dfTrain, dfStores, on="Store", how="inner")
dfNew = dfNew.fillna(0)


dfNew['DayDate'] = dfNew['Date'].apply(lambda x: x.split("-")[2])
dfNew['MonthDate'] = dfNew['Date'].apply(lambda x: x.split("-")[1])
dfNew['YearDate'] = dfNew['Date'].apply(lambda x: x.split("-")[0])

storeType = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
dfNew['StoreType'] = [storeType[item] for item in dfNew['StoreType']]

assortment = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
dfNew['Assortment'] = [assortment[item] for item in dfNew['Assortment']]

plt.scatter(dfNew['Sales'], dfNew['Customers'], alpha=0.5)
plt.show()

print(dfNew.corr())
dfNew


### FUNCTIONS USED IN DATA PRE PROCESSING
def procMonthPromo(x):
    if x == '0':
        return '0'
    names = x.split(",")
    nums = []
    for n in names:
        if n == "Jan":
            nums.append("1")
        if n == "Feb":
            nums.append("2")
        if n == "Mar":
            nums.append("3")
        if n == "Apr":
            nums.append("4")
        if n == "May":
            nums.append("5")
        if n == "Jun":
            nums.append("6")
        if n == "Jul":
            nums.append("7")
        if n == "Aug":
            nums.append("8")
        if n == "Sept":
            nums.append("9")
        if n == "Oct":
            nums.append("10")
        if n == "Nov":
            nums.append("11")
        if n == "Dec":
            nums.append("12")
    
    numbers = ",".join(nums)
    return numbers


def promoDay(df):
    ret = 0
    if df['PromoMonths'] == '0':
        return 0
    
    numbers = df['PromoMonths'].split(',')
    
    for n in numbers:
        if int(df['MonthDate']) == int(n):
            ret = 1
            break
    
    return ret

def competitionTime(df):
    if df['CompetitionOpenSinceMonth'] == 0 and df['CompetitionOpenSinceYear'] == 0:
        return 0
    
    m = 12 - df['CompetitionOpenSinceMonth']
    y = (df['YearDate'] - df['CompetitionOpenSinceYear'] - 1) * 12
    if y < 0:
        y = 0
    m = m + y + df['MonthDate']
    return m

##### PRE PROCESSING
def dataPreProcessing(csvFile):
    dfTrain = pd.read_csv(csvFile)

    dfStores = pd.read_csv('lojas.csv')

    dfNew = pd.merge(dfTrain, dfStores, on="Store", how="inner")
    dfNew = dfNew.fillna(0)

    dfNew['DayDate'] = dfNew['Date'].apply(lambda x: int(x.split("-")[2]))
    dfNew['MonthDate'] = dfNew['Date'].apply(lambda x: int(x.split("-")[1]))
    dfNew['YearDate'] = dfNew['Date'].apply(lambda x: int(x.split("-")[0]))

    if csvFile == "dataset_treino.csv":
        dfNew = dfNew[dfNew['Open'] == 1]

    storeType = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    dfNew['StoreType'] = [storeType[item] for item in dfNew['StoreType']]

    assortment = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    dfNew['Assortment'] = [assortment[item] for item in dfNew['Assortment']]

    dfNew['PromoMonths'] = dfNew['PromoInterval'].apply(procMonthPromo)

    dfNew['PromoDay'] = dfNew.apply(promoDay, axis=1)

    dfNew = dfNew.convert_objects(convert_numeric=True)

    dfNew['CompetitionTime'] = dfNew.apply(competitionTime, axis=1)



    return dfNew
    


######## TESTS AREA

### MODEL 02 -- POLY REGRESSION ** BEST
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

dfOpen = dataPreProcessing("dataset_treino.csv")
dfOpen

x = ["Open", "Promo", "DayOfWeek", "DayDate", "MonthDate", "YearDate", "SchoolHoliday", "StoreType", "Assortment", "Promo2",
    "CompetitionDistance", "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear"]
y = "Sales"

x_train, x_test, y_train, y_test = train_test_split(dfOpen[x], dfOpen[y], test_size=0.20, random_state=101)
    
model = GradientBoostingRegressor(loss='lad', max_depth=5,
                                max_features=None,
                                min_samples_leaf=6,
                                min_samples_split=6,
                                n_estimators=500)
print("Fiting")
model.fit(x_train, y_train)
print("Predicting")
prediction = model.predict(x_test)

    
result = pd.DataFrame(columns=['Test', 'Prediction'])
result['Test'] = y_test
result['Prediction'] = prediction
print('MAE Model 02:', metrics.mean_absolute_error(result['Test'],result['Prediction']))




#### SUBMISSION AREA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

#dfOpen = dataPreProcessing("dataset_treino.csv")
dfOpen = dataPreProcessing("dataset_treino.csv")
x = ["Open", "Promo", "DayOfWeek", "DayDate", "MonthDate", "YearDate", 
    "SchoolHoliday", "StoreType", "Assortment", "Promo2", "PromoDay", "CompetitionTime"]
y = "Sales"

model = GradientBoostingRegressor(loss='lad', max_depth=5,
                                max_features=None,
                                min_samples_leaf=6,
                                min_samples_split=6,
                                n_estimators=500)


print("Fiting")
model.fit(dfOpen[x], dfOpen[y])
print("Finish training")

#dfTest = dataPreProcessing("dataset_teste.csv")
dfTest = dataPreProcessing("dataset_teste.csv")    
prediction = model.predict(dfTest[x])
    
result = pd.DataFrame(columns=['Id', 'Sales'])
result['Id'] = dfTest['Id']
result['Sales'] = prediction * dfTest['Open']
print("Finish Prediction")


result.to_csv("subGra.csv", index=False)
