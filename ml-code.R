#===============================================================================
#                           ENVIRONMENT SETUP
#===============================================================================


# Clean history
rm(list = ls())

# Load libraries
require(readxl)
require(caret)
require(randomForest)
require(glmnet)
require(class)
require(naivebayes)
require(rpart)
require(rpart.plot)
require(gridExtra)
require(pROC)
require(vip)
require(cluster)
require(clustvarsel)
require(mclust)
require(ggcorrplot)
require(klaR)
require(vscc)
require(pgmm)
require(NbClust)
require(dbscan)
require(ClustOfVar)
require(corrgram)



# Fetch data
df1 = read_excel(file.choose(), sheet = 1)
df2 = read_excel(file.choose(), sheet = 2)

#===============================================================================
# DATA ENGINEERING
#===============================================================================
View(df1)
View(df2)

str(df1) # 3,195 x 54
str(df2) # 24,611 x 8



# some data pre-processing before merging datasets
any(is.na(df1$area_name))
df1[which(is.na(df1$state_abbreviation)), ]
df1 = df1[(df1$area_name != 'United States'),] # remove USA title (not needed)

# each state has a unique identifier (state_abbreviation)
states = df1[(is.na(df1$state_abbreviation)), ]
dim(states)[[1]] # 51 states

# create a separate column stating the state of each record
df1$state = NULL
state_lookup = c(
  AL = "Alabama", AK = "Alaska", AZ = "Arizona", AR = "Arkansas",
  CA = "California", CO = "Colorado", CT = "Connecticut", DE = "Delaware",
  FL = "Florida", GA = "Georgia", HI = "Hawaii", ID = "Idaho",
  IL = "Illinois", IN = "Indiana", IA = "Iowa", KS = "Kansas",
  KY = "Kentucky", LA = "Louisiana", ME = "Maine", MD = "Maryland",
  MA = "Massachusetts", MI = "Michigan", MN = "Minnesota", MS = "Mississippi",
  MO = "Missouri", MT = "Montana", NE = "Nebraska", NV = "Nevada",
  NH = "New Hampshire", NJ = "New Jersey", NM = "New Mexico", NY = "New York",
  NC = "North Carolina", ND = "North Dakota", OH = "Ohio", OK = "Oklahoma",
  OR = "Oregon", PA = "Pennsylvania", RI = "Rhode Island", SC = "South Carolina",
  SD = "South Dakota", TN = "Tennessee", TX = "Texas", UT = "Utah",
  VT = "Vermont", VA = "Virginia", WA = "Washington", WV = "West Virginia",
  WI = "Wisconsin", WY = "Wyoming", DC = "District of Columbia"
)

df1$state = state_lookup[df1$state_abbreviation]
df1 = df1[is.na(df1$state_abbreviation) == FALSE, ]

# create separate column referring to the counties
df1$county = NULL # initialize
df1$county = sub(" \\S+$", "", df1$area_name)
length(df1$county) == dim(df1)[[1]] # true
View(df1[, c(2,56)])
df1$area_name = NULL

length(unique(df1$state)) # 51 states
length(unique(df2$state)) # 49 states
setdiff(df1$state_abbreviation, df2$state_abbreviation)
# states with abbreviation "DC", "MN" are not in df2

# something i noticed
df1$county[df1$state_abbreviation=="AK"]
df2$county[df2$state_abbreviation=="AK"]
# The two datasets use incompatible geographic divisions for Alaska. In df1,
# Alaska data is organized by county-equivalents (Boroughs and Census Areas) with
# standard FIPS codes. However, df2 uses State House Districts (political subdivisions).
# Since these boundaries do not align, there is no 1:1 mapping between the two
# datasets, leading to failed matches and NA values during the merge despite the
# shared 'AK' state abbreviation. Thus, i am removing alaska from the analysis.
# it is usually treated separataley anyways.
df1 = df1[df1$state_abbreviation!="AK",]
df2 = df2[df2$state_abbreviation!="AK",]

# MERGE df1 to df2 with respect to fips (True unique variable) and state abbr
data = merge(df2, df1, by = c("fips", "state_abbreviation")) # inner join to keep only rows that match on both dataframes
length(unique(data$state_abbreviation)) # 40 states

lost_states = setdiff(df2$state_abbreviation, data$state_abbreviation)
lost_states
# In total,9 states were excluded from the analysis: Alaska, due to incompatible
# geographic subdivisions, and CT, KS, MA, ME, MN, ND, NH, RI, and VT, which
# reported primary results at the congressional/legislative district level
# rather than the county level, making a county-level merge impossible.

all(data$state.x == data$state.y, na.rm = TRUE) # TRUE
all(data$county.x == data$county.y, na.rm = TRUE) # FALSE
# SINCE county.x ≠ county.y, it’s almost certainly:
# "St Louis" vs "St. Louis"
# "Baltimore" vs "Baltimore City"

# more inspection
subset(data, county.x != county.y)[, c("fips", "county.x", "county.y")]
# confirms that i merged correctly, but there are some name missmatches
# e.g. county.x : Saint Francis, county.y : St. Francis
# so i'll keep only the county.x
# for the state.x and state.y, since theyre all the same, we dont mind what we keep

# delete redundant variables
data$county.y = NULL
data$state.y = NULL
# rename needed ones
names(data)[names(data) == "county.x"] = "county"
names(data)[names(data) == "state.x"] = "state"

View(data)
str(data) # 17,479 x 59

colSums(is.na(data)) # all good

save(data, file = "datamerged.RData")
load("datamerged.RData")



#---- some basic eda

# check for missing data
any(is.na(data)) # FALSE
# check some basic descriptives
summary(data) # everything seems ok, no anomalies detected
# check for any duplicates
sum(duplicated(data)) # 0 duplicates



#-------------------------------------------------------------------------------
# TRUMP SUBSET
#-------------------------------------------------------------------------------
# filter to trump rows only
trump = data[data$candidate == 'Donald Trump', ]
# create binary variable stating whether trump got more than 50% or less
# 1 binary outcome: did Trump exceed 50% of the Republican primary vote in that county?
str(trump) # 2,711 x 59

trump$trump_majority = NULL
trump$trump_majority = as.factor(ifelse(trump$fraction_votes > 0.50, 1, 0))
levels(trump$trump_majority) = c('lte50', 'gt50')
# lte50: less than or equal to 50%
# gt50: greater than 50%
table(trump$trump_majority)
# lte50  gt50
# 1714   997

# visualize
t = table(trump$trump_majority)
tt = prop.table(t)
custom_labels = c("Less than or equal to 50%", "Greater than 50%")
b = barplot(t,
        col = c("grey", "indianred3"),
        border = "white",
        main = "Trump Votes Frequencies",
        xlab = "Classes",
        names.arg = custom_labels,
        ylim = c(0, 2000))
labels_text = paste0(t, " (", round(tt, 2) * 100, "%)")
text(x = b, y = t, label = labels_text, pos = 3, cex = 0.9, col = "black", font=2)

# This is the 2016 Republican primary, which is a very different thing.
# In the primary, Trump was competing against many other Republican candidates
# (Cruz, Rubio, Kasich, Carson, etc.) simultaneously. With so many candidates
# splitting the vote, it was actually very common for Trump to win a state/county
# without getting 50%+ — you just need a plurality (more votes than anyone else),
# not a majority. For example, let's look at Autauga county from our data:
#
# Trump: 0.445 (44.5%) -> wins the county
# Cruz: 0.205
# Rubio: 0.148
# Carson: 0.146
# Kasich: 0.035
#
# Trump wins Autauga with the most votes, but still has trump_majority = lte50
# because he didn't crack 50%. This is completely normal and expected.
# We know that on the 2016 US elections, trump won.
# Trump won the primary overall, but in the majority of individual counties he
# did so with a plurality rather than a majority.
# I've researched all of this because i had no clue how the US election system works..



# then remove the others but KEEP state
trump = trump[, !(names(trump) %in% c("fips", "state",
                                      "state_abbreviation",
                                      "county",
                                      "candidate",
                                      "party", "votes",
                                      "fraction_votes"))]

save(trump, file = "trumpdata.RData")
load("trumpdata.RData")

#--- some basic EDA
par(mfrow = c(7, 8), mar = c(3, 3, 2, 1))
for(col in names(trump)[-52]){ # exclude last variable (trump_majority)
  hist(trump[[col]],
       main = col,
       xlab = "Value",
       col = "blue",
       border = "white")}
dev.off()

par(mfrow = c(7, 8), mar = c(3, 3, 2, 1))
for(col in names(trump)[-52]){ # exclude last variable (trump_majority)
  boxplot(trump[[col]],
          main = col,
          xlab = "Value",
          col = "blue")}
dev.off()


numeric = trump[,-52]
par(mfrow = c(7, 8), mar = c(3, 3, 2, 1))
for (col in names(numeric)){
  boxplot(trump[[col]] ~ trump$trump_majority,
          main = col,
          col = c("grey50", "indianred3"))}
dev.off()

par(mfrow = c(7, 8), mar = c(2, 2, 2, 1))
for (col in names(numeric)){
  gt50_mean = mean(trump[[col]][trump$trump_majority == "gt50"], na.rm = TRUE)
  lte50_mean = mean(trump[[col]][trump$trump_majority == "lte50"], na.rm = TRUE)
  barplot(c(gt50_mean, lte50_mean),
          names.arg = c("gt50", "lte50"),
          main = col,
          col = c("grey50", "indianred3"),
          border = "white")}

dev.off()




#===============================================================================
# TRAINING / TESTING SPLIT
#===============================================================================
set.seed(222)
index = createDataPartition(trump$trump_majority, p = 0.7, list = FALSE)
train_data = trump[index, ] ; dim(train_data)[[1]]
test_data  = trump[-index, ] ; dim(test_data)[[1]]

# Reorder so gt50 is the first level (positive class)
# Positive class (1) = the outcome we're trying to detect/predict
# Negative class (0) = the baseline/reference
train_data$trump_majority = factor(train_data$trump_majority,
                                    levels = c("gt50", "lte50"))

test_data$trump_majority  = factor(test_data$trump_majority,
                                    levels = c("gt50", "lte50"))
# set-up the 10-fold CV
control = trainControl(method = "cv",
                       number = 10,
                       classProbs = TRUE,
                       summaryFunction = twoClassSummary)
# this enables the AUC evaluation, and probability predictions

#-------------------------------------------------------------------------------
# MODEL 1 : LOGISTIC REGRESSION (with Regularization (Lasso/Ridge))
#-------------------------------------------------------------------------------

# Using my training data, find the best Elastic Net model to predict the target
# variable. Try 100 different combinations of hyperparameter settings, evaluate
# them using the cross-validation rules I set in control, and give me the model
# that achieves the best ROC score. glmnet also internally scales the variables
set.seed(222)
model1 = train(
  trump_majority ~ .,
  data = train_data,
  method = "glmnet",
  trControl = control,
  metric = "ROC",
  tuneLength = 10
)

model1$bestTune
model1$results # for all 100 trained models
model1$results[rownames(model1$bestTune), ]
#      alpha    lambda       ROC      Sens      Spec      ROCSD     SensSD    SpecSD
# 91     1 3.397218e-05 0.7440066 0.4241822 0.8691667 0.04190622 0.06763277 0.0364408
logit_probs = predict(model1,test_data, type = "prob")
head(round(logit_probs,2))

logit_pred = predict(model1, test_data)
confusionMatrix(logit_pred, test_data$trump_majority) # Accuracy : 0.7159
#            Reference
# Prediction gt50 lte50
# gt50       132    64
# lte50      167   450

#-------------------------------------------------------------------------------
# MODEL 2: SIMPLE LINEAR REGRESSION
#-------------------------------------------------------------------------------
# TO PREVENT LEAKAGE WHEN RE-RUNNING OTHER MODELS:
set.seed(222)
train_data2 = train_data
test_data2 = test_data
train_data2$trump_majority_num = ifelse(train_data$trump_majority == "gt50", 1, 0)
test_data2$trump_majority_num  = ifelse(test_data$trump_majority == "gt50", 1, 0)

control_lm = trainControl(method = "cv", number = 10)

set.seed(222)
model2 = train(
  trump_majority_num ~ . - trump_majority,
  data = train_data2,
  method = "lm",
  trControl = control_lm)

model2$results
# intercept      RMSE  Rsquared       MAE     RMSESD RsquaredSD      MAESD
#      TRUE 0.4535762 0.1470302 0.4008151 0.01984984 0.04176469 0.01280358
lm_pred_probs = predict(model2, newdata = test_data2)
# general rule :
# if yi > 0.5 --> assign to class 1
# if yi < 0.5 --> assign to class 0
lm_pred_class = ifelse(lm_pred_probs > 0.5, "gt50", "lte50")
lm_pred_class = factor(lm_pred_class, levels = levels(test_data$trump_majority))
confusionMatrix(lm_pred_class, test_data2$trump_majority) # Accuracy : 0.7085

#           Reference
# Prediction gt50 lte50
#      gt50   113    51
#      lte50  186   463

#-------------------------------------------------------------------------------
# MODEL 3: K-NN
#-------------------------------------------------------------------------------
set.seed(222)
model3 = train(trump_majority ~ .,
                data = train_data,
                method = "knn",
                preProcess = c("zv", "center", "scale"), # ESSENTIAL to scale for k-NN
                trControl = control,
                metric = "ROC",
                tuneLength = 10) # to tune the number of nearest neighbours (k)
model3$bestTune # 17 neighbours
model3$results
model3$results[rownames(model3$bestTune),]
#    k       ROC      Sens Spec      ROCSD     SensSD     SpecSD
# 7 17 0.7579819 0.3798551 0.92 0.04575757 0.06549021 0.03267876
knn_preds = predict(model3, newdata = test_data)
confusionMatrix(knn_preds, test_data$trump_majority) # Accuracy : 0.6765

#            Reference
# Prediction gt50 lte50
# gt50       86    50
# lte50      213   464
#-------------------------------------------------------------------------------
# MODEL 4: NAIVE BAYES CLASSIFIER
#-------------------------------------------------------------------------------
set.seed(222)
model4 = train(trump_majority ~ .,
               data = train_data,
               method = "naive_bayes",    # use the 'naivebayes' package
               preProcess = c("center", "scale"),
               trControl = control,
               metric = "ROC")

model4$bestTune
model4$results
model4$results[rownames(model4$bestTune), ]
# usekernel laplace adjust       ROC     Sens      Spec      ROCSD     SensSD     SpecSD
#  TRUE       0      1 0.6618434 0.339648 0.8391667 0.05865357 0.04899263 0.07027833
nb_preds = predict(model4, test_data)
confusionMatrix(nb_preds, test_data$trump_majority) # Accuracy : 0.6531

#             Reference
# Prediction gt50 lte50
# gt50        87    70
# lte50      212   444

#-------------------------------------------------------------------------------
# MODEL 5: DECISION TREE
#-------------------------------------------------------------------------------
set.seed(222)
model5 = train(
  trump_majority ~ .,
  data = train_data,
  method = "rpart",        # rpart = recursive partitioning (standard decision tree)
  trControl = control,     # existing 10-fold CV setup
  metric = "ROC",
  tuneLength = 10          # tunes the complexity parameter (cp)
)
model5$bestTune # cp = 0.008595989
model5$results
model5$results[rownames(model5$bestTune),]
#        cp       ROC      Sens      Spec      ROCSD     SensSD     SpecSD
# 0.008595989 0.7042219 0.4900207 0.8483333 0.03382283 0.05032621 0.04406084

# Get probabilities
tree_probs = predict(model5, newdata = test_data, type = "prob")

# Find optimal cutoff
roc_obj = roc(test_data$trump_majority, tree_probs[,"gt50"])
k = coords(roc_obj, "best", best.method = "youden", transpose = FALSE)
custom_cutoff = k$threshold[1]
custom_cutoff # 0.349

# Apply it
tree_preds = ifelse(tree_probs[,"gt50"] > custom_cutoff, "gt50", "lte50")
tree_preds = factor(tree_preds, levels = c("gt50", "lte50"))

# Evaluate
confusionMatrix(tree_preds, test_data$trump_majority) # accuracy : 0.69

#             Reference
# Prediction gt50 lte50
# gt50        133    86
# lte50       166   428

# Plot
# Extract the actual rpart model from caret
fit = model5$finalModel

# dentify which column in yval2 represents the 'gt50' probability
# Usually, caret orders levels alphabetically. If 'gt50' is your target:
# Column 2 = lte50 prob, Column 3 = gt50 prob (approx, based on rpart structure)
gt50_probs = fit$frame$yval2[, 3]

# Apply custom 0.39 threshold
# If the probability of gt50 is >= 0.39, we set the label to 2 ('gt50')
# Otherwise, we set it to 1 ('lte50')
fit$frame$yval = ifelse(gt50_probs >= 0.349, 2, 1)

# Plot the modified model
prp(fit, type = 4, extra = 101, box.palette = list("RdBu", "RdPu"),
    main = "Trump Majority Tree (Cutoff: 0.349)", cex = 0.5)



#-------------------------------------------------------------------------------
# MODEL 6: RANDOM FORREST
#-------------------------------------------------------------------------------
set.seed(222)
model6 = train(
  trump_majority ~ .,
  data = train_data,
  method = "rf",
  trControl = control,
  metric = "ROC",
  tuneLength = 5
)

model6$bestTune
model6$results
model6$results[rownames(model6$bestTune),]
# mtry      ROC      Sens Spec      ROCSD     SensSD     SpecSD
#  26 0.828611 0.5960455 0.89 0.04310135 0.05075157 0.04808044

rf_preds = predict(model6, newdata = test_data)
confusionMatrix(rf_preds, test_data$trump_majority) # Accuracy : 0.7478

#             Reference
# Prediction gt50 lte50
# gt50       158    64
# lte50      141   450

#-------------------------------------------------------------------------------
# MODEL 7: LDA
#-------------------------------------------------------------------------------
set.seed(222)
model7 = train(
  trump_majority ~ .,
  data = train_data,
  method = "lda",          # no tuning parameters
  trControl = control,
  preProcess = c("center"),
  metric = "ROC"
)
model7$results
# parameter       ROC      Sens  Spec      ROCSD     SensSD     SpecSD
#      none 0.7368145 0.3855487 0.905 0.04386039 0.07799836 0.02670137

lda_preds = predict(model7, newdata = test_data)
confusionMatrix(lda_preds, test_data$trump_majority) # Accuracy : 0.7122

#            Reference
# Prediction gt50 lte50
# gt50        118    53
# lte50       181   461

#-------------------------------------------------------------------------------
# MODEL 8: QDA
#-------------------------------------------------------------------------------
set.seed(222)
model8 = train(
  trump_majority ~ .,
  data = train_data,
  method = "qda",
  trControl = control,
  preProcess = c("center"),
  metric = "ROC"
)
model8$results
# parameter       ROC      Sens      Spec      ROCSD     SensSD    SpecSD
#        none 0.7383024 0.8065424 0.4883333 0.06063565 0.08222067 0.1683343

qda_preds = predict(model8, newdata = test_data)
confusionMatrix(qda_preds, test_data$trump_majority) # Accuracy : 0.6458

#            Reference
# Prediction gt50 lte50
# gt50       262   251
# lte50       37   263
#-------------------------------------------------------------------------------
# MODEL 9: SVM (Support Vector Machines)
#-------------------------------------------------------------------------------
set.seed(222)
model9 = train(
  trump_majority ~ .,
  data = train_data,
  method = "svmRadial",        # radial basis function kernel (most common)
  preProcess = c("center", "scale"),  # ESSENTIAL... SVMs are sensitive to scale
  trControl = control,
  metric = "ROC",
  tuneLength = 10              # tunes Cost (C) and sigma automatically
)
model9$results
model9$bestTune                # shows best C and sigma values sigma = 1/2gamma!!!!
model9$results[rownames(model9$bestTune), ]
# sigma      C       ROC      Sens      Spec      ROCSD     SensSD     SpecSD
# 0.03209286 4 0.8005052 0.5559627 0.8816667 0.02764313 0.04165169 0.04509934


svm_preds = predict(model9, newdata = test_data)
confusionMatrix(svm_preds, test_data$trump_majority) # accuracy = 0.7306


#===============================================================================
# OVERALL MODEL COMPARISON
#===============================================================================
# get probabilities for each model
logit_probs = predict(model1, test_data, type = "prob")
knn_probs = predict(model3, test_data, type = "prob")
nb_probs = predict(model4, test_data, type = "prob")
rf_probs = predict(model6, test_data, type = "prob")
lda_probs = predict(model7, test_data, type = "prob")
qda_probs = predict(model8, test_data, type="prob")
svm_probs = predict(model9, newdata = test_data, type = "prob")

# build ROC objects
roc1 = roc(test_data$trump_majority, logit_probs[,"gt50"])
roc3 = roc(test_data$trump_majority, knn_probs[,"gt50"])
roc4 = roc(test_data$trump_majority, nb_probs[,"gt50"])
roc5 = roc(test_data$trump_majority, tree_probs[,"gt50"])
roc6 = roc(test_data$trump_majority, rf_probs[,"gt50"])
roc7 = roc(test_data$trump_majority, lda_probs[,"gt50"])
roc8 = roc(test_data$trump_majority, qda_probs[,"gt50"])
roc9 = roc(test_data$trump_majority, svm_probs[, "gt50"])

# plot all together
plot(roc1, col = "blue", lwd = 2, main = "ROC Curve Comparison")
plot(roc3, col = "red", lwd = 2, add = TRUE)
plot(roc4, col = "green", lwd = 2, add = TRUE)
plot(roc5, col = "orange", lwd = 2, add = TRUE)
plot(roc6, col = "purple", lwd = 2, add = TRUE)
plot(roc7, col = "brown", lwd = 2, add = TRUE)
plot(roc8, col = "pink2", lwd = 2, add = TRUE)
plot(roc9, col = "yellow", lwd = 2, add = TRUE)

legend("bottomright",
       legend = c(paste("Logistic AUC =", round(auc(roc1), 3)),
                  paste("KNN AUC =", round(auc(roc3), 3)),
                  paste("Naive Bayes =", round(auc(roc4), 3)),
                  paste("Tree AUC =", round(auc(roc5), 3)),
                  paste("RF AUC =", round(auc(roc6), 3)),
                  paste("LDA AUC =", round(auc(roc7), 3)),
                  paste("QDA AUC =", round(auc(roc8), 3)),
                  paste("SVM AUC =", round(auc(roc9), 3))),
       col = c("blue","red","green","orange","purple","brown", "pink2", "yellow"),
       lwd = 2, bty="n")

#-------------------------------------------------------------------------------
accuracies = c(0.7159, 0.7122, 0.6458, 0.7478, 0.6765, 0.69, 0.6531, 0.7085,0.7306)
models = c("Logistic", "LDA", "QDA", "Random Forest", "KNN", "Decision Tree", "Naive Bayes", "Linear Regression", "SVM")

ord = order(accuracies)
accuracies = accuracies[ord]
models = models[ord]


par(mar = c(4, 8, 3, 4))
bp = barplot(accuracies, names.arg = models,
             horiz = TRUE, col = "seashell4",
             border = NA, xlim = c(0, 1),
             xlab = "Accuracy", main = "Model Accuracy Comparison",
             las = 1, cex.names = 0.9, font=2)
abline(v = 0.5, lty = 2, col = "grey10", lwd = 2)
text(accuracies + 0.02, bp,
     labels = paste0(round(accuracies * 100, 2), "%"),
     cex = 0.85, font=2, adj = 0)
#===============================================================================
# FEATURE IMPORTANCE
#===============================================================================
# ggplot-based varImp plots
vimp_plot = function(model, title) {
  vi = varImp(model)
  ggplot(vi, top = 10) + ggtitle(title)
}

p1 = vimp_plot(model1, "Logistic Regression")
# variable importance plot is not compatible with simple linear regression model (its not supported)
p3 = vimp_plot(model3, "KNN")
p4 = vimp_plot(model4, "Naive Bayes")
p5 = vimp_plot(model5, "Decision Tree")
p6 = vimp_plot(model6, "Random Forest")
p7 = vimp_plot(model7, "LDA")
p8 = vimp_plot(model8, "QDA")
p9 = vimp_plot(model9, "SVM")
# arrange all in one window (3 columns x 3 rows)
grid.arrange(p1, p3, p4, p5, p6, p7, p8, p9, ncol = 4)


#===============================================================================
#                                   PART 2
#===============================================================================
# Here we want to use the "demographic related" variables to cluster the counties
# and then use  the "economic related" to describe the clusters you have found.
# We will use MODEL BASED CLUSTERING and incorporate hierarchical clustering also
# if i have time -- DO DBSCAN


# create a new dataframe that will be used for clustering and only keep
# the demographic and economic variables (which are all numeric)
k = df1[!(df1$state_abbreviation %in% c("NA", NA)), ]
df = k[, !(names(df1) %in% c("fips", "state_abbreviation",
                               "state", "county", "area_name"))]
head(df)

demo = c("PST045214", "PST040210", "PST120214", "POP010210",
         "AGE135214", "AGE295214", "AGE775214", "SEX255214",
         "RHI125214", "RHI225214", "RHI325214", "RHI425214",
         "RHI525214", "RHI625214", "RHI725214", "RHI825214",
         "POP715213", "POP645213", "POP815213", "EDU635213",
         "EDU685213", "VET605213")

eco = c("LFE305213", "HSG010214", "HSG445213", "HSG096213", "HSG495213",
        "HSD410213", "HSD310213", "INC910213", "INC110213", "PVY020213",
        "BZA010213", "BZA110213", "BZA115213", "NES010213", "SBO001207",
        "SBO315207", "SBO115207", "SBO215207", "SBO515207", "SBO415207",
        "SBO015207", "MAN450207", "WTN220207", "RTN130207", "RTN131207",
        "AFN120207", "BPS030214", "LND110210", "POP060210")

demo_df = df[, demo] # cluster with those
eco_df = df[, !names(df) %in% demo] # interpret clusters with those
length(names(demo_df)) + length(names(eco_df)) == length(names(df)) # true - all good

corrgram(demo_df)
#===============================================================================
# --- VARIABLE SELECTION:
#-------------------------------------------------------------------------------
# Correlation based feature selection with clustering for high dimensional data
#-------------------------------------------------------------------------------
#===============================================================================
nzv = nearZeroVar(demo_df)
df_clean = if (length(nzv) > 0) demo_df[, -nzv] else demo_df
df_clean = as.data.frame(df_clean)

tree = hclustvar(X.quanti = df_clean)
plot(tree, main = "Feature Dendrogram")

set.seed(222)
stab = stability(tree, B = 50)

k = 4
# Plot the dendrogram with the cut line
plot(tree, main = "Feature Dendrogram (cut at k=4)")
rect.hclust(tree, k = 4, border = "red")


k = 7
plot(tree, main = "Feature Dendrogram (cut at k=7)")
rect.hclust(tree, k=7, border = "blue")

clusters = cutreevar(tree, k=7)

# Pick the variable with highest squared loading per cluster
repr_vars = sapply(clusters$var, function(cl) {
  rownames(cl)[which.max(cl[, "squared loading"])]})
print(repr_vars)


# cluster1    cluster2    cluster3    cluster4    cluster5    cluster6    cluster7
# "POP010210" "EDU685213" "AGE295214" "SEX255214" "RHI125214" "RHI625214" "POP815213"

# so now we will keep only the subset
demo_new = demo_df[, c("POP010210", "EDU685213", "AGE295214", "SEX255214",
                       "RHI125214", "RHI625214", "POP815213")]
head(demo_new)

# now scale the data to proceed with the clustering
demo_scaled = scale(demo_new)
head(demo_scaled)


#-------------------------------------------------------------------------------
# DISTANCE BASED CLUSTERING --> HIERARCHICAL CLUSTERING
#-------------------------------------------------------------------------------
# distance matrix
dist_matrix = dist(demo_scaled, method = "euclidean")
# find optimal number of clusters
set.seed(222)
nb_ward = NbClust(demo_scaled,
                  distance = "euclidean",
                  min.nc = 2,          # minimum clusters to try
                  max.nc = 6,          # maximum clusters to try
                  method = "ward.D2",  # linkage method
                  index = "all")      # try all 30 indices
# *******************************************************************
# * Among all indices:
# * 6 proposed 2 as the best number of clusters
# * 4 proposed 3 as the best number of clusters
# * 4 proposed 4 as the best number of clusters
# * 1 proposed 5 as the best number of clusters
# * 6 proposed 6 as the best number of clusters
#
# ***** Conclusion *****
#
# * According to the majority rule, the best number of clusters is  2
#
#
# *******************************************************************

# Try all different linkage methods, see whats best
linkage_methods = c("ward.D2", "complete", "average", "single", "centroid")

par(mfrow = c(2, 3))
for (method in linkage_methods){
  hc = hclust(dist_matrix, method = method)
  plot(hc,
       labels = FALSE,
       main = paste("Linkage:", method),
       xlab = "",
       sub = "")
}
dev.off()
# Ward Linkage wins!

# fit hierarchical clustering with Ward linkage
set.seed(222)
hc = hclust(dist_matrix, method = "ward.D2")

# agglomerative coefficient
agnes(demo_scaled, method = "ward")$ac # 0.9914479 --> 1 really good

# lets do some height plots to confirm the selection of k = 5 from the previous algorithm

par(mfrow=c(1,2))
heights = rev(hc$height)
plot(1:20, heights[1:20],
     type = "b",
     xlab = "Number of clusters",
     ylab = "Merge height",
     main = "Height Plot")

# elbow starts at k=2 indeed

plot(hc$height,
     main = "Height Plot",
     xlab = "Index",
     ylab = "Height",
     pch = 21)
# indeed k=2 seems to be the best


# plot dendrogram (ward linkage) and cut it at k
par(mfrow=c(2,2))

plot(hc, labels=FALSE,
     main = "Dendrogram (k=2)",
     xlab="", sub="")
rect.hclust(hc, k=2, border="red")

plot(hc, labels=FALSE,
     main = "Dendrogram (k=3)",
     xlab="", sub="")
rect.hclust(hc, k=3, border="red")

plot(hc,
     labels = FALSE,
     main   = "Dendrogram (k=4)",
     xlab   = "",
     sub    = "")
rect.hclust(hc, k = 4 , border = "red")

plot(hc,
     labels = FALSE,
     main   = "Dendrogram (k=5)",
     xlab   = "",
     sub    = "")
rect.hclust(hc, k = 5 , border = "red")

# silhouette plots
set.seed(222)
par(mfrow = c(2, 2))
clusters_2 = cutree(hc, k = 2)
sil_2 = silhouette(clusters_2, dist_matrix)
plot(sil_2,
     main = "Silhouette (Ward) | k=2",
     border = NA, col="grey60")

clusters_3 = cutree(hc, k = 3)
sil_3 = silhouette(clusters_3, dist_matrix)
plot(sil_3,
     main = "Silhouette (Ward) | k=3",
     border = NA, col="grey60")

clusters_4 = cutree(hc, k = 4)
sil_4 = silhouette(clusters_4, dist_matrix)
plot(sil_4,
     main = paste("Silhouette (Ward) | k = 4"),
     border = NA, col = "grey60")

clusters_5 = cutree(hc, k = 5)
sil_5 = silhouette(clusters_5, dist_matrix)
plot(sil_5,
     main = paste("Silhouette (Ward) | k = 5"),
     border = NA, col="grey60")
dev.off()
# we will keep 2 clusters
# explained in report thouroughly


# add cluster labels to eco_df and explain the clusters using economic variables
eco_df$HC = NA
eco_df$HC = as.factor(cutree(hc, k = 3))
# mean of economic variables per cluster
aggregate(. ~ HC, data = eco_df, FUN = mean, na.rm = TRUE)
# boxplots
cluster_colors = c("violet", "lightseagreen", "orange")

par(mfrow = c(5, 6))
for (var in eco){
  boxplot(eco_df[[var]] ~ eco_df$HC,
          main = var,
          xlab = "Cluster",
          ylab = "",
          col  = cluster_colors)
}
dev.off()

#-------------------------------------------------------------------------------
# for report
par(mfrow = c(2, 5))
selected_vars = c("LFE305213", "HSG445213", "HSG096213", "HSG495213",
                  "INC910213", "INC110213", "PVY020213", "HSD310213",
                  "SBO015207", "SBO415207")

for (var in selected_vars){
  boxplot(eco_df[[var]] ~ eco_df$HC,
          main = var,
          xlab = "Cluster",
          ylab = "",
          col  = cluster_colors)
}
dev.off()


selected = c("LFE305213", "HSG445213", "HSG096213", "HSG495213",
              "INC910213", "INC110213", "PVY020213", "HSD310213",
              "SBO015207", "SBO415207", "HC")
res = aggregate(. ~ HC, data = eco_df[, selected], FUN = median, na.rm = TRUE)
round(res[, -1], 2)

#-------------------------------------------------------------------------------
dev.off()
#-------------------------------------------------------------------------------
# MODEL BASED CLUSTERING --> GAUSSIAN MIXTURE MODEL
#-------------------------------------------------------------------------------
set.seed(222)
gmm_icl = mclustICL(demo_scaled, G = 2:5)
summary(gmm_icl)
# Best ICL values:
#               VVV,5      VEV,5      VVV,4
# ICL      -33337.47 -34624.977 -35126.107
# ICL diff      0.00  -1287.502  -1788.632
plot(gmm_icl)

# look at silhouettes and cluster sizes
# Compute silhouette for each k
par(mfrow = c(2, 2))
set.seed(222)
for (k in 2:5){
  gmm_k = Mclust(demo_scaled, G = k)
  sil = silhouette(gmm_k$classification, dist(demo_scaled))
  plot(sil, main = paste("GMM Silhouette k =", k),
       col = 1:k, border = NA)
  avg_sil = mean(sil[, 3])}

# i will keep 2 clusters..
set.seed(222)
gmm2 = Mclust(demo_scaled, G = 2)
summary(gmm2)


# add cluster labels to eco_df and describe clusters
eco_df$GMM = NA
eco_df$GMM = as.factor(gmm2$classification)
# mean economic variables per cluster
aggregate(. ~ GMM, data = eco_df, FUN = mean, na.rm = TRUE)
# boxplots
par(mfrow = c(5, 6))
for (var in eco){
  boxplot(eco_df[[var]] ~ eco_df$GMM,
          main = var,
          xlab = "Cluster",
          ylab = "",
          col  = c("orchid", "lightsalmon"))}
dev.off()

pairs(demo_scaled, col = eco_df$GMM,
     pch = 19, cex = 0.3, main = "Gaussian Mixture Model")

#-------------------------------------------------------------------------------
# DENSITY BASED CLUSTERING --> DBSCAN
#-------------------------------------------------------------------------------
minPts = 8

# k-NN distance plot to find eps
kNNdistplot(demo_scaled, k = minPts)
abline(h = 0.5, col = "purple", lty =2)
abline(h = 1, col = "red", lty = 2)
abline(h = 1.5, col = "green4", lty = 2)

# For each point, it plots the distance to its k-th nearest neighbor, sorted from
# smallest to largest. Flat for most points -> everyone is equally dense.
# Sudden spike only at the very end -> only a few outliers are far away.
# No clear elbow in the middle -> no density gap between groups
# BASICALLY
# Elbow in the middle: Multiple meaningful clusters
# Elbow at the very end: One big cluster + few outliers (which is what we have)

# Try different eps values
set.seed(222)
for (eps in c(0.5, 0.75, 1.0, 1.5)){
  db = dbscan(demo_scaled, eps = eps, minPts = 8)
  cat("eps =",eps, ", Clusters:", max(db$cluster),
      ", Noise:", sum(db$cluster == 0),
    "(", round(mean(db$cluster == 0) * 100, 1), "%)\n")}


# Let's go with eps=1.5 as it gives 2 clusters with acceptable noise (4.4%)
set.seed(222)
db = dbscan(demo_scaled, eps = 1.5, minPts = 8)
table(db$cluster)
# Cluster 0 (noise): 137 points (4.4%)
# Cluster 1: 2970 points (95% of our data!)
# Cluster 2: only 7 points

# This is not a meaningful clustering.Cluster 2 is essentially just 7 outliers
# that happened to be dense enough to form a micro-cluster, and everything else
# got dumped into one giant cluster.
# This happens because our data is one big continuous dense blob with no natural
# density-based separation, which the k-NN plot already told us.

