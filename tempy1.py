
from sklearn import ensemble
from sklearn.metrics import r2_score
X = training_feature_df
y = training_df["Fare"]
X2 = dup_testing_feature_df
y2 = dup_testing_df["Fare"]

X3 = testing_feature_df


# for gamma in gamma_values:
#     gamma = float(gamma)/100
# gamma = 0.21 # 1e3
# gamma = 0.26 # 1e4

max_depths = range(3, 10, 2),
min_child_weights = range(2, 6, 2)
gammas = [(i+1)/10.0 for i in range(0, 5)]
subsamples = [i/10.0 for i in range(6, 10)]
n_estimatorss = range(1000, 7000, 1000)

for max_depth in max_depths:
    for min_child_weight in min_child_weights:
        for gamma in gammas:
            for subsample in subsamples:
                for n_estimators in n_estimatorss:
                    params = {'n_estimators': n_estimators, 'max_depth': max_depth,
                              'min_samples_split': min_child_weight, 'learning_rate': gamma, 'loss': 'ls', 'subsample': subsample}
                    clf = ensemble.GradientBoostingRegressor(**params)
                    clf.fit(X, y)

                    predictions = clf.predict(X2)
                    # for i, actual in enumerate(dup_testing_df["Fare"]):
                    #     print "Actual " + str(actual) + ", Predicted " + str(predictions[i])
                    # print lm.score(X2, y2)
                    print(max_depth, min_child_weight, gamma, subsample, n_estimators, r2_score(
                        dup_testing_df["Fare"], predictions))

# actual_predictions = clf.predict(X3)
# print([p for p in actual_predictions])
