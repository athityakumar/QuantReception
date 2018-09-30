import urllib2
import pandas as pd
import datetime

# training_data_file = "https://s3-ap-southeast-1.amazonaws.com/mettl-arq/questions/codelysis/machine-learning/fare-prediction/train.csv"
training_data_file = "file:///Volumes/Nanosuit/train.csv"
testing_data_file = "file:///Volumes/Nanosuit/test.csv"
training_data = urllib2.urlopen(training_data_file).read()
testing_data = urllib2.urlopen(testing_data_file).read()

training_df_data = []
testing_df_data = []

for line in training_data.split("\n")[:-1]:
    columns = line.rstrip().split(",")
    training_df_data.append(columns)

for line in testing_data.split("\n")[:-1]:
    columns = line.rstrip().split(",")
    testing_df_data.append(columns)

training_df = pd.DataFrame(
    data=training_df_data[1:6001], columns=training_df_data[0])
dup_testing_df = pd.DataFrame(
    data=training_df_data[6001:], columns=training_df_data[0])
testing_df = pd.DataFrame(
    data=testing_df_data[1:6001], columns=testing_df_data[0])

SALUTATIONS = ["Dr.", "Mr.", "Mrs.", "Miss"]
CLASSES = ["Economy", "Business"]
CITIES = ['Mumbai', 'Lucknow', 'Kolkata',
          'Chennai', 'Delhi', 'Patna', 'Hyderabad']

print training_df
print testing_df


def str_to_date(datestr):
    yyyy, mm, dd = datestr.split("-")
    return datetime.date(int(yyyy), int(mm), int(dd))


def hot_encode(index, length):
    encoding = [1 if index == i else 0 for i in range(length)]
    return encoding


def flatten_list(l):
    flat_list = []
    for sublist in l:
        if isinstance(sublist, list):
            for item in sublist:
                flat_list.append(item)
        else:
            flat_list.append(sublist)
    return flat_list


def fetch_feature_map(df):
    feature_data = []
    # feature_columns = ["Salutation", "Age", "Days of booking before journey", "Class", "From", "To", "Day of booking", "Day of journey", "Time"]

    for row in df.iterrows():
        ref = row[1]
        salutation = ref["Name"].split(" ")[0]
        salutation_index = SALUTATIONS.index(salutation)
        salutation_index = hot_encode(salutation_index, len(SALUTATIONS))
        age = 2018 - int(ref["Date of Birth"].split("-")[0])
        n_days = str_to_date(ref["Flight Date"]) - \
            str_to_date(ref["Booking Date"])
        n_days = 0 if str(n_days) == '0:00:00' else int(
            str(n_days).split(" ")[0])
        class_index = hot_encode(CLASSES.index(ref["Class"]), len(CLASSES))
        from_index = hot_encode(CITIES.index(ref["From"]), len(CITIES))
        to_index = hot_encode(CITIES.index(ref["To"]), len(CITIES))
        booking_day = hot_encode(str_to_date(ref["Booking Date"]).weekday(), 7)
        journey_day = hot_encode(str_to_date(ref["Flight Date"]).weekday(), 7)
        time = int(ref["Flight Time"].split(":")[0])

        # print flatten_list(list([salutation_index, age, n_days, class_index, from_index, to_index, booking_day, journey_day, time]))

        feature_data.append(flatten_list(list([salutation_index, float(
            age)/100, n_days/100, class_index, from_index, to_index, booking_day, journey_day, float(time)/24])))
        #=> 0.64

    feature_df = pd.DataFrame(feature_data)
    return feature_df


training_feature_df = fetch_feature_map(training_df)
dup_testing_feature_df = fetch_feature_map(dup_testing_df)
testing_feature_df = fetch_feature_map(testing_df)

# from sklearn import linear_model

# X = training_feature_df[list(training_feature_df.columns)]
# y = training_df["Fare"]
# model = sm.OLS(y, X).fit()
# predictions = model.predict(X)
# model.summary()

# X = training_feature_df
# y = training_df["Fare"]
# lm = linear_model.LinearRegression()
# model = lm.fit(X, y)

# X2 = testing_feature_df
# y2 = testing_df["Fare"]
# predictions = lm.predict(X2)
# for i, actual in enumerate(testing_df["Fare"]):
#     print "Actual " + str(actual) + ", Predicted " + str(predictions[i])
# print lm.score(X2, y2)
# print([p for p in predictions])

from sklearn.metrics import r2_score
import xgboost
X = training_feature_df
y = training_df["Fare"]

X2 = dup_testing_feature_df
y2 = dup_testing_df["Fare"]

X3 = testing_feature_df


clf = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)
clf.fit(X, y)
predictions = clf.predict(X2)
print(r2_score(dup_testing_df["Fare"], predictions))

# actual_predictions = clf.predict(X3)
# print([p for p in actual_predictions])
