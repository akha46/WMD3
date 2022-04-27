import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

all_data = pd.read_csv("vineyard_weather_1948-2017.csv",low_memory=False)
all_data[["year", "month", "day"]] = all_data['DATE'].str.split("-", expand= True)
week_list = []
count = 1
for ind in all_data.index:
    date_value = datetime.date(int(all_data["year"][ind]), int(all_data["month"][ind]), int(all_data["day"][ind]))
    week_number = date_value.isocalendar()[1]
    week_list.append(week_number)
all_data["week_number"] = week_list
all_data["storm"] = all_data.apply(lambda row : True if (row.PRCP >= 0.35 and row.TMAX <= 80 ) else False , axis =1)
selected_col = all_data.drop(columns=["DATE","month","day"])
selected_col =  selected_col[selected_col["week_number"]>=35]
selected_col =  selected_col[selected_col["week_number"]<=40]
selected_col.replace({False: 0, True: 1}, inplace=True)
week_split = selected_col.groupby(['year','week_number'])[['PRCP','TMAX','TMIN','storm']].agg(['mean'])

week_split['storm'] = week_split['storm'].apply(np.ceil)
week_split['storm'] = week_split['storm'].replace([week_split['storm'] > 0.0], 1)

week_split_x = week_split[['PRCP','TMAX','TMIN']].values
week_split_y = week_split['storm'].values
all_x = pd.DataFrame(week_split_x, columns = ['PRCP','TMAX','TMIN'])
all_y = pd.DataFrame(week_split_y, columns = ['storm'])

X_train, X_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
dtc_model = clf.fit(X_train, y_train)
y_pred = dtc_model.predict(X_test)
print("Accuracy")
print(accuracy_score(y_test, y_pred))
print('Prescision')
print(precision_score(y_test, y_pred))
print('Recall')
print(recall_score(y_test, y_pred))
print('\n')
cmd = confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity_d = tn / (tn+fp)
print(' specificity: %0.5f' % specificity_d )
sensitivity_d = tp / (tp + fn)
print(' sensitivity: %0.5f' % sensitivity_d )

def calc_payout(p_s, specificity_d, cost_h, cost_nh_s, cost_nh_ns):
    p_dns_ns = specificity_d*(1- p_s)
    p_dns = specificity_d*(1-p_s) + (1-specificity_d)*(p_s)
    p_ns_dns = p_dns_ns/p_dns
    e_val = p_dns*(cost_nh_ns*p_ns_dns + cost_nh_s*(1-p_ns_dns)) + cost_h*(1-p_dns)
    return e_val



chance_mold = st.number_input('Chance of botrytis  mold in percent')

chance_ns = st.number_input('Chance of no sugar level grape in percent')

chance_ts = st.number_input('Chance of typical sugar level grape in percent')

chance_hs = st.number_input('Chance of high sugar level grape in percent')

cost_h =12*(6000*5+2000*10+2000*15)
Cost_NH_NS = 12*(6000*5 +2000*10 + 2000*15)
P_NH_NS = chance_ns/100
Cost_NH_TS = 12*(5000*5 + 1000*10 + 2500*15 + 1500*30)
P_NH_TS = chance_ts/100
Cost_NH_HS = 12*(4000*5 + 2500*10 + 2000*15 +1000*30 + 500*40)
P_NH_HS = chance_hs/100

Cost_NH_M = 12*(5000*5 + 1000*10 +2000*120)
P_NH_M = chance_mold/100
Cost_NH_NM = 12*(5000*5 + 1000*10)
P_NH_NM = 1-P_NH_M
p_s = 0.5

cost_nh_s = max((P_NH_NM * Cost_NH_NM + P_NH_M * Cost_NH_M), cost_h)
cost_nh_ns = max((P_NH_NS *Cost_NH_NS + P_NH_TS*Cost_NH_TS + P_NH_HS *Cost_NH_HS),cost_h)
estimate = calc_payout(0.5,specificity_d,cost_h,cost_nh_s,cost_nh_ns)
clair = cost_h - estimate


def generate_evalue(prob_storm, sensitivity, specificity, p_harvest, p_wait_ns_no_sugar, p_wait_ns_typical_sugar,
                    p_wait_ns_high_sugar, p_wait_storm_mold, p_wait_storm_no_mold, p_mold, p_no_sugar, p_typical_sugar,
                    p_high_sugar):
    val_ns_harvest = p_harvest
    val_ns_s = 0.14 * (p_mold * p_wait_storm_mold + (1 - p_mold) * p_wait_storm_no_mold)
    val_ns_ns = 0.3 * (
            p_no_sugar * p_wait_ns_no_sugar + p_typical_sugar * p_wait_ns_typical_sugar + p_high_sugar * p_wait_ns_high_sugar)
    val_ns = 0.7 * max(val_ns_harvest, val_ns_s, val_ns_ns)

    val_s_harvest = p_harvest
    val_s_s = 0.67 * (p_mold * p_wait_storm_mold + (1 - p_mold) * p_wait_storm_no_mold)
    val_s_ns = 0.33 * (
            p_no_sugar * p_wait_ns_no_sugar + p_typical_sugar * p_wait_ns_typical_sugar + p_high_sugar * p_wait_ns_high_sugar)
    val_s = 0.3 * max(val_s_harvest, val_s_s, val_s_ns)
    val_wait = val_s + val_ns
    return val_wait


e_val = generate_evalue(prob_storm=0.5, sensitivity=0.12, specificity=0.9,
                        p_harvest=960000,
                        p_wait_ns_no_sugar=960000, p_wait_ns_typical_sugar=1410000,
                        p_wait_ns_high_sugar=1500000,
                        p_wait_storm_mold=3300000, p_wait_storm_no_mold=420000,
                        p_mold=P_NH_M,
                        p_no_sugar=P_NH_NS, p_typical_sugar=P_NH_TS,
                        p_high_sugar=P_NH_HS
                        )
if(cost_h<e_val):
    st.title("Choose Clairvoyance")
else:
    st.title("No Clairvoyance")