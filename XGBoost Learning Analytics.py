import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2 as pg2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from xgboost import XGBClassifier, plot_importance

#Reads SQL table of Student Information
secret = '1234'
conn = pg2.connect(database='ou_la', user = 'postgres', password = secret)
cur = conn.cursor()
cur.execute('SELECT * FROM studentInfo')
print(cur.fetchmany(3))
df = pd.read_sql_query("SELECT * FROM studentInfo", conn)

#Exploratory Stage
print(df.head())
print(df.shape)
print(df.describe())

pres_modules = {
    'AAA-2013J': 0,
    'AAA-2014J': 0,
    'BBB-2013B': 1,
    'BBB-2013J': 2,
    'BBB-2014B': 1,
    'BBB-2014J': 2,
    'CCC-2014B': 3,
    'CCC-2014J': 4,
    'DDD-2013B': 5,
    'DDD-2013J': 6,
    'DDD-2014B': 5,
    'DDD-2014J': 6,
    'EEE-2013J': 7,
    'EEE-2014B': 8,
    'EEE-2014J': 7,
    'FFF-2013B': 9,
    'FFF-2013J': 10,
    'FFF-2014B': 9,
    'FFF-2014J': 10,
    'GGG-2013J': 11,
    'GGG-2014B': 12,
    'GGG-2014J': 11
}

imd_dic = {
        '0-10%': 1,
        '10-20%': 2,
        '20-30%': 3,
        '30-40%': 4,
        '40-50%': 5,
        '50-60%': 6,
        '60-70%': 7,
        '70-80%': 8,
        '80-90%': 9,
        '90-100%': 10,
    }

df['student_and_module'] = df['id_student'].astype(str) + df['code_module']
df['module_and_presentation'] = df['code_module'] + "-" + df['code_presentation']
df['pres_groups'] = df['module_and_presentation'].map(pres_modules)
df['dep_dec'] = df['imd_band']
df['dep_dec'] = df['dep_dec'].map(imd_dic)

#Only keep students' first attempt
df_earliest = df.drop_duplicates(subset = 'student_and_module', keep = 'first')
df_first = df_earliest[df_earliest['number_of_prev_attempts'] == 0]

df_first['distinction'] = np.where(df_first['final_result'] == 'Distinction', 1, 0)
print(df_first['distinction'].sum())
print((df_first['pres_groups'] == 2).sum())

#One Hot Code Categorical Variables
df_first_OHE = pd.get_dummies(df_first, columns = ['gender', 'highest_education', 'disability', 'age_band', 'region'], prefix = ['gender', 'highest_education', 'disability', 
                                                                                                                                 'age_band', 'region'])

## Weighted variance, skewness and kurtosis
def calculate_weighted_moments(group):
    days = group['click_date']
    clicks = group['sum_click']
    
    # Weighted mean
    weighted_mean = (days * clicks).sum() / clicks.sum()

    # Weighted variance
    weighted_variance = ((clicks * (days - weighted_mean)**2).sum() / clicks.sum())

    if weighted_mean == 0:
        coef_of_var = np.nan  # Set CV to NaN if weighted mean is zero
    else:
        coef_of_var = (weighted_variance ** 0.5) / weighted_mean

    # Check for zero or NaN variance
    if weighted_variance == 0 or pd.isna(weighted_variance):
        # Return NaN for skewness and kurtosis if variance is zero
        return pd.Series({
            'click_time_variance': weighted_variance,
            'click_time_skewness': np.nan,
            'click_time_kurtosis': np.nan,
            'click_time_cv': np.nan
        })

    weighted_skewness = ((clicks * (days - weighted_mean)**3).sum() / (clicks.sum() * (weighted_variance ** (3/2))))

    # Weighted kurtosis
    weighted_kurtosis = ((clicks * (days - weighted_mean)**4).sum() / (clicks.sum() * weighted_variance**2)) - 3

    return pd.Series({
        'click_time_variance': weighted_variance,
        'click_time_skewness': weighted_skewness,
        'click_time_kurtosis': weighted_kurtosis,
        'click_time_cv': coef_of_var
    })


sVle = pd.read_sql_query("SELECT * FROM studentVle", conn)
moments = sVle.groupby(['id_student', 'code_module', 'code_presentation']).apply(calculate_weighted_moments)
print(moments.head())

print(moments)

moments = moments.reset_index()

df_with_moments = pd.merge(df_first_OHE, moments, on=['id_student', 'code_module', 'code_presentation'], how='left')

# Check the merged DataFrame
print(df_with_moments.head())

print(df_first_OHE.columns.tolist())

X = df_with_moments[['studied_credits', 'total_clicks_url', 'total_clicks_forumng', 'total_clicks_homepage', 'total_clicks_oucontent', 'total_clicks_subpage', 
                'total_clicks_resource', 'total_clicks_sharedsubpage', 'total_clicks_page', 'total_clicks_questionnaire', 'total_clicks_ouwiki', 'total_clicks_htmlactivity', 
                'total_clicks_ouelluminate', 'total_clicks_dataplus', 'total_clicks_externalquiz', 'total_clicks_repeatactivity', 'total_clicks_dualpane', 'total_clicks_quiz', 
                'total_clicks_glossary', 'total_clicks_oucollaborate', 'total_clicks_folder', 'pres_groups', 'dep_dec',
                'distinction', 'gender_F', 'gender_M', 'highest_education_A Level or Equivalent', 'highest_education_HE Qualification', 'highest_education_Lower Than A Level', 
                'highest_education_No Formal quals', 'highest_education_Post Graduate Qualification', 'disability_N', 'disability_Y', 'age_band_0-35', 'age_band_35-55', 
                'age_band_55<=', 'region_East Anglian Region', 'region_East Midlands Region', 'region_Ireland', 'region_London Region', 'region_North Region', 
                'region_North Western Region', 'region_Scotland', 'region_South East Region', 'region_South Region', 'region_South West Region', 'region_Wales', 
                'region_West Midlands Region', 'region_Yorkshire Region', 'click_time_variance', 'click_time_skewness', 'click_time_kurtosis', 'click_time_cv']]

X.columns=X.columns.str.replace(' ', '_')
X.columns=X.columns.str.replace('<=', '+')
X.columns=X.columns.str.replace(',', '')

#For each model

precisions = []
recalls = []
F1s = []
for i in range(1, 13):
    X_temp = X[X['pres_groups'] == i].copy()
    X_temp = X_temp.loc[:, (X_temp != 0).any(axis=0)]
    print(X.head())
    y = X_temp['distinction']
    X_temp = X_temp.drop(columns = ['distinction', 'pres_groups'])
    X_train, X_test, Y_train, y_test = train_test_split(X_temp, y, test_size=0.2, random_state=1)

    model = XGBClassifier(use_label_encoder = False, 
                    eval_metric = 'logloss', 
                    scale_pos_weight = 9, 
                    max_depth = 3,
                    subsample = 0.5)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    #brief attempt at changing decision threshold
    y_pred = (model.predict_proba(X_test)[:,1] >= 0.3).astype(int)
    print(classification_report(y_test, y_pred))
    F1s.append(f"F1 scores for module type {i} = {tuple(f1_score(y_test, y_pred, average=None))}")
    precisions.append(f"Precision scores for module type {i} = {tuple(precision_score(y_test, y_pred, average=None))}")
    recalls.append(f"Recall scores for module type {i} = {tuple(recall_score(y_test, y_pred, average=None))}")
    print(precisions)
    print(recalls)
    print(F1s)    
    plt.rcParams["figure.figsize"] = (18, 9)
    plot_importance(model, title=f"Feature Importance - module presentation {i}", max_num_features = 15, importance_type = "gain", values_format = "{v:.2f}")
    plt.show()

#Over all modules:
y = X['distinction']
X = X.drop(columns = ['distinction', 'pres_groups'])
X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = XGBClassifier(use_label_encoder = False, 
                    eval_metric = 'logloss', 
                    scale_pos_weight = 2, 
                    max_depth = 3,
                    subsample = 0.5)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
y_pred = (model.predict_proba(X_test)[:,1] >= 0.3).astype(int)
print(classification_report(y_test, y_pred))
precisions.append(f"Precision scores for all modules = {tuple(precision_score(y_test, y_pred, average=None))}")
recalls.append(f"Recall scores for all modules = {tuple(recall_score(y_test, y_pred, average=None))}")
F1s.append(f"F1 score over all modules = {tuple(f1_score(y_test, y_pred, average=None))}")
print(precisions)
print(recalls)
print(F1s)
plt.rcParams["figure.figsize"] = (18, 9)
plot_importance(model, title=f"Feature Importance - across all modules", importance_type = "gain", max_num_features = 15, xlabel = "Gain per variable", 
                ylabel = "Top 15 Features", values_format = "{v:.2f}")
plt.show()






#Top 10 Students (dormant for now)
if False:
    ed_dic = {
            'No Formal quals': 0,
            'Lower Than A Level': 0,
            'A Level or Equivalent': 0,
            'HE Qualification': 1,
            'Post Graduate Qualification': 1
        }
    df_with_moments['ed_dummy'] = df_with_moments['highest_education']
    df_with_moments['ed_dummy'] = df_with_moments['ed_dummy'].map(ed_dic)
    print(df_with_moments['ed_dummy'])
    
    
    top_dic = {}
    deg_list = []
    for i in range(13):
        scores_filtered = df_with_moments[df_with_moments['pres_groups'] == i]
        overall_mean = np.mean(scores_filtered['ed_dummy'])
        top_10_avg_scores = scores_filtered['avg_score'].nlargest(10) 
        top_ten = scores_filtered[scores_filtered['avg_score'].isin(top_10_avg_scores)]
        pres_mean = np.mean(top_ten['ed_dummy'])
        top_dic[i] = (pres_mean, overall_mean)
        if i != 2 and i != 4:
            deg_list.append(pres_mean)
            print(top_ten)
    print(top_dic)
    print(deg_list)
    total = (100 * round(np.mean(deg_list), 4)), (100 * round(np.mean(df_with_moments['ed_dummy']), 4))
    
    plt.style.use('seaborn')
    fig, axs = plt.subplots(3,4, figsize = (12,8))
    plt.gca().set_facecolor('lightblue')
    count = 0
    for i, ax in enumerate(axs.flat):
        if count < len(top_dic):
            values = top_dic[count]
            print(values)
            values_perc = (100 * values[0], 100 * values[1])
            ax.set_facecolor('lightblue')
            ax.bar(['% With \nDegree', '% of Module with \nDegree'], values_perc, color = ['blue', 'green'])
            title = list(pres_modules.keys())[count]
            ax.set_title(title)
            ax.set_ylim(0,100)
            count += 1
            if count == 2 or count == 4:
                count += 1
    values = total
    ax.set_ylim(0,100)
    ax.bar(['% With \nDegree', '% of Module with \nDegree'], total, color = ['gold', 'silver'])
    plt.tight_layout()
    plt.show()
    
    
    
    
