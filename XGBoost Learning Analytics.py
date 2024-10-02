import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2 as pg2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
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

#Merge 2013 and 2014 instances by year half
mod_pres_dic = {
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
df['pres_groups'] = df['module_and_presentation'].map(mod_pres_dic)
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

print(df_first_OHE.columns.tolist())

X = df_first_OHE[['studied_credits', 'total_clicks_url', 'total_clicks_forumng', 'total_clicks_homepage', 'total_clicks_oucontent', 'total_clicks_subpage', 
                  'total_clicks_resource', 'total_clicks_sharedsubpage', 'total_clicks_page', 'total_clicks_questionnaire', 'total_clicks_ouwiki', 'total_clicks_htmlactivity', 
                  'total_clicks_ouelluminate', 'total_clicks_dataplus', 'total_clicks_externalquiz', 'total_clicks_repeatactivity', 'total_clicks_dualpane', 'total_clicks_quiz', 
                  'total_clicks_glossary', 'total_clicks_oucollaborate', 'total_clicks_folder', 'pres_groups', 'dep_dec',
                  'distinction', 'gender_F', 'gender_M', 'highest_education_A Level or Equivalent', 'highest_education_HE Qualification', 'highest_education_Lower Than A Level', 
                  'highest_education_No Formal quals', 'highest_education_Post Graduate Qualification', 'disability_N', 'disability_Y', 'age_band_0-35', 'age_band_35-55', 
                  'age_band_55<=', 'region_East Anglian Region', 'region_East Midlands Region', 'region_Ireland', 'region_London Region', 'region_North Region', 
                  'region_North Western Region', 'region_Scotland', 'region_South East Region', 'region_South Region', 'region_South West Region', 'region_Wales', 
                  'region_West Midlands Region', 'region_Yorkshire Region']]

X.columns=X.columns.str.replace(' ', '_')
X.columns=X.columns.str.replace('<=', '+')
X.columns=X.columns.str.replace(',', '')

#For each model
F1s = []
CRs = []
for i in range(1, 13):
    X_temp = X[X['pres_groups'] == i].copy()
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
    print(classification_report(y_test, y_pred))
    F1s.append(f"F1 scores for module type {i} = {tuple(f1_score(y_test, y_pred, average=None))}")
    print(F1s)
    plot_importance(model)
    plt.show()

#Over all modules:
if False:
    print(X.head())
    y = X['distinction']
    X = X.drop(columns = ['distinction', 'pres_groups'])
    X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model = XGBClassifier(use_label_encoder = False, 
                        eval_metric = 'logloss', 
                        scale_pos_weight = 3, 
                        max_depth = 4,
                        subsample = 0.5)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    F1s.append(f"F1 score over all modules = {tuple(f1_score(y_test, y_pred, average=None))}")
    print(F1s)





if False:
        #Get just instances where module = module_presentation

        #Only include resources which are actually present in that module

        #Train Model

        #Predict Model

        #Feature Importance


    #Show Graph of Feature Importance


    #Top 10 Students
    top_dic = {}
    deg_list = []
    for i in range(13):
        scores_filtered = scores[scores['pres_groups'] == i]
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
    total = (100 * round(np.mean(deg_list), 4)), (100 * round(np.mean(scores['ed_dummy']), 4))

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
            title = list(mod_pres_dic.keys())[count]
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




