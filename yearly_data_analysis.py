import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

os.chdir(r'C:\Users\josep\python_files\CptS_575\project')

#====================================================================================================
#  S&P 500
#====================================================================================================
sp = pd.read_csv(r'./data/additions.csv')
sp = sp[(sp.index_name.str.contains('S&P 500 Comp-Ltd'))]
sp['add_date'] = sp['addition_date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
sp['deletion_date'] = sp['deletion_date'].fillna(20211231.0)
sp['del_date'] = pd.to_datetime(sp['deletion_date'], format='%Y%m%d.0', errors='coerce')
sp.head()
sp['cusip'] = sp['cusip'].astype('str')
sp['cusip'] = sp.cusip.str[:-1]

# get additions data
additions = sp[(sp['add_date'] >= '1990-1-1 01:00:00')]
sp_add_counts = additions.groupby(['year']).id.count()
sp_add_counts.plot()
plt.ylim([0,55])
plt.title('S&P 500 Additions')
plt.ylabel('Stock Additions in Year')
plt.xlabel('Year')
plt.savefig(r'./figures/Additions.png')

# get current s&p 500 members and additions for year t
def get_current_sp(sp, year):
    current_sp = sp[(sp['year'] <= year-1) & ~(sp['del_date'].dt.year <= year-1)]
    additions = sp[(sp['year'] == year)]
    return current_sp, additions

sp_2000, sp_2000_add = get_current_sp(sp, 2000)

#====================================================================================================
#  Stock Data
#====================================================================================================
df = pd.read_stata(r'./data/monthly_stock_ind.dta')
df['abs_prc'] = df.prc.abs()
# backward fill prc
#df['abs_prc'] = df['abs_prc'].bfill()

df['market_cap'] = df['abs_prc']*df['shrout']

#len(set(df[df.market_cap > 4000000].cusip))


# create date variables
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

#len(set(df[df['market_cap'] > 1000000].cusip))

# reshape the dataframe
# take only year end
# should i take the log of everything
df_end = df[df['month'] == 12]

# take last (non-na) occurence of siccd, ticker, primexch, and comnam
#fillers = df_end.groupby(['cusip'])['siccd', 'ticker', 'primexch', 'comnam'].last().reset_index()
#fillers = df_end.groupby(['cusip'])['siccd', 'ticker', 'primexch', 'comnam'].nth(-1).reset_index()
fillers = df_end.groupby(['cusip'])['siccd', 'ticker', 'primexch', 'comnam'].max().reset_index()
fillers.columns = [fillers.columns[0]] + [col+'_filler' for col in fillers.columns[1:]]

#merge in filler
#df_end = df_end.merge(fillers, how='left', on='cusip')

#get rid of duplicates by aggregating
df_end = df_end.sort_values(['cusip', 'year', 'market_cap', 'abs_prc', 'shrout', 'ret', 'divamt'], ascending=[False for _ in range(7)]).drop_duplicates(['cusip', 'year'])

#t = df_end.pivot(index='cusip', columns='year', values=['abs_prc'])
stock = df_end.pivot(index='cusip', columns='year', values=['abs_prc','market_cap', 'shrout'])#.reset_index()
stock.columns = pd.Index([e[0] + '_' + str(e[1]) for e in stock.columns])
stock = stock.reset_index()
t_stock = stock.merge(fillers, how='left', on='cusip')


filler_value = .001
t_stock = t_stock.fillna(filler_value)

num_cols = [col for col in t_stock if 'filler' not in col]
num_cols.remove('cusip')

# reshape data from wide to long
stubnames = ['abs_prc', 'market_cap', 'shrout']
t_stock_long = pd.wide_to_long(t_stock, stubnames, i='cusip', j='year', sep='_').reset_index()

log_stubnames = ['log_' + col for col in stubnames]
t_stock_long[log_stubnames] = np.log(t_stock_long[stubnames])

# take the difference of the logs
diff_stubnames = ['diff_'+col for col in stubnames]
t_stock_long[diff_stubnames] = t_stock_long.sort_values(['cusip', 'year']).groupby('cusip')[log_stubnames].diff()

# reshape back to wide
idx = [col for col in t_stock_long.columns].index('abs_prc')
t_stock_wide = t_stock_long.pivot(index='cusip', columns='year', values=t_stock_long.columns[idx:])#.reset_index()
t_stock_wide.columns = pd.Index([e[0] + '_' + str(e[1]) for e in t_stock_wide.columns])
t_stock_wide = t_stock_wide.reset_index()
stock = t_stock_wide.merge(fillers, how='left', on='cusip')

t = df_end.describe()
t['market_cap'] = t['market_cap']/1000
t['shrout'] = t['shrout']/1000
print(t[['abs_prc', 'shrout', 'market_cap']].to_latex())

#====================================================================================================
#  add indicators for sp500
#====================================================================================================

#def remove_current_sp(stock, sp_current):
#    t_stock_wide.cusip.isin(sp_current.cusip)
#    temp[temp['_merge'] != 'both']
#    return temp.drop('_merge')
#
#def add_sp_ind(stock, sp_add):
#    temp = stock.merge(sp_add, how='left', on='cusip', indicator=True)

for year in range(1990,2019):
    sp_current, sp_add = get_current_sp(sp, year)
    stock['current_sp_'+str(year)] = stock['cusip'].isin(sp_current['cusip'])
    stock['add_sp_'+str(year)] = stock['cusip'].isin(sp_add['cusip'])

stock['exchange'] = pd.Categorical(stock.primexch_filler)
stock['exchange'] = stock['exchange'].cat.codes

#====================================================================================================
#  Run RandomForestClassifier on data (for now just 2000)
#====================================================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

year = 2000
f1_scores_means = []
likely_df = pd.DataFrame()

for preda in [.15]:#[.05, .075, .1,  .2, .25, .3, .35, .4]:
    accuracies = []
    recalls = []
    precisions = []
    f1_scores = []
    for year in range(1994,2018):
        print(year)
        #train on 2000 - first take out current_sp_2000 == True
        X = stock[stock['current_sp_'+str(year)] == False]
        #X = stock
        
        #drop vars that end with 1990
        
        
        #drop obs with current market cap not good
        X = X[X['market_cap_'+str(year-1)] > 1000000]
        #X = X[X['market_cap_'+str(year)] > 1000000]
        
        #drop add_sp_ vars and keep only one
        y = X['add_sp_'+str(year)]
        #y = X['current_sp_'+str(year)]
        
        #drop years after current year
        drop_year_cols = []
        for yr in range(year, 2019):
            cols = [col for col in X if col.endswith(str(yr))]
            drop_year_cols += cols
            
        drop_cols = [col for col in X if col.startswith('current_sp_') or col.startswith('add_sp_')] + ['cusip'] + [col for col in X if col.endswith('_filler')] + [col for col in X if col.endswith('_1990')] + drop_year_cols
        temp = X.copy()
        X = X.drop(columns=drop_cols)
        X_whole = X.copy()
        #
        #rf = RandomForestClassifier()
        #rf.fit(X,y)
        #
        #preds = rf.predict(X)
        #np.mean(preds == y)
        
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=1)
        
        #==================================================================================
        # Upsampling
        
        X = pd.concat([X_train, y_train], axis=1)
    
        # separate minority and majority classes
        not_fraud = X[X['add_sp_'+str(year)]==0]
        fraud = X[X['add_sp_'+str(year)]==1]
        
        # upsample minority
        fraud_upsampled = resample(fraud,
                                  replace=True, # sample with replacement
                                  n_samples=len(not_fraud), # match number in majority class
                                  random_state=27) # reproducible results
        
        # combine majority and upsampled minority
        upsampled = pd.concat([not_fraud, fraud_upsampled])
        
        X_train = upsampled.drop(columns=['add_sp_'+str(year)])
        y_train = upsampled['add_sp_'+str(year)]
        
        #
        #===================================================================================
        
        
        
        #random forest
        rf = RandomForestClassifier()
        rf.fit(X_train,y_train)
        preds = pd.Series(rf.predict_proba(X_test)[:,1])
        preds_real = rf.predict(X_test)
        np.mean(preds_real == y_test)
        print(confusion_matrix(preds_real, y_test))
        #preds = rf.predict(X_test)
        
        
        
        from sklearn.metrics import confusion_matrix
        #confusion_matrix(y_test, preds)
        
        scores = pd.concat([preds, y_test.reset_index()], axis=1).drop(columns='index')
        scores.columns=['y_pred', 'y_true']
        scores.sort_values('y_true')
        scores['new_pred'] = scores.y_pred > preda
        acc = np.mean(scores['y_true'] == scores['new_pred'])
        accuracies.append(acc)
        rec = recall_score(scores['y_true'], scores['new_pred'])
        recalls.append(rec)
        prec = precision_score(scores['y_true'], scores['new_pred'])
        precisions.append(prec)
        f1 = f1_score(scores['y_true'], scores['new_pred'])
        f1_scores.append(f1)
        #print(acc)
        #print(confusion_matrix(scores['y_true'], scores['new_pred']))
        pred_idx = X.reset_index()[scores['new_pred']].index
        year_list = [year for _ in range(len(pred_idx))]
        
        
        
        
        rf = RandomForestClassifier()
        rf.fit(X_whole,y)
        preds = pd.Series(rf.predict_proba(X_whole)[:,1])
        preds_real = rf.predict(X_whole)
        print(confusion_matrix(preds_real, y))
        
        pred_idx = temp.reset_index().loc[preds_real, ['cusip',f'diff_abs_prc_{year+1}', f'diff_market_cap_{year+1}']]
        pred_idx.columns = ['cusip','diff_abs_prc', f'diff_market_cap']
        pred_idx['year'] = year
        pred_idx['add'] = 0
        y_idx = temp.loc[y, ['cusip',f'diff_abs_prc_{year+1}', f'diff_market_cap_{year+1}']]
        y_idx.columns = ['cusip','diff_abs_prc', f'diff_market_cap']
        y_idx['year'] = year
        y_idx['add'] = 1
        
        likely_y = pd.concat([pred_idx, y_idx], ignore_index=True)
        likely_df = pd.concat([likely_df, likely_y], ignore_index=True)
        
    f1_scores_means.append(np.mean(f1_scores))


likely_df.to_stata(r'treat_control.dta')

years = range(1991, 2018)
df = pd.concat([pd.Series(df) for df in [years, accuracies, recalls, precisions]], axis=1)
df.columns = ['year', 'accuracy', 'recall', 'precision']


df = df.set_index('year')
df.T.to_csv(r'results.csv', index=True)

#
#
#
#
#rf = RandomForestClassifier()
#rf.fit(X_whole,y)
#preds = pd.Series(rf.predict_proba(X_whole)[:,1])
#preds_real = rf.predict(X_whole)
#print(confusion_matrix(preds_real, y))



#
#t = df_end[['date', 'comnam', 'abs_prc']].pivot(index='comnam', columns='date', values='abs_prc')
#
#sns.heatmap(t_stock.isnull(), cbar=False)
#sns.heatmap(stock.isnull(), cbar=False)
#
#t = df_end[['date', 'comnam', 'abs_prc']].set_index(['date', 'comnam'])
#t.unstack(1)



