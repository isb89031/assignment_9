
##################################
##################################
# IMPORT LIBRARIES AND LOAD DATA
##################################
##################################

library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3tuning)

# You could read the titanic dataset that was saved in the Python version
#t2 <- read.csv("./data/titanic_openml.csv")  #your file path may be different
#str(t2)
#head(t2)
#summary(t2)

# We can also try to load from OpenML
#install.packages("mlr3oml")
library(mlr3oml)
list_oml_data_sets(data_name = "titanic")  #search for titanic datasets
odata <- OMLData$new(id=40945)  #get titanic dataset from OpenML using the data_id
t2 <- as.data.frame(odata$data)
str(t2)
head(t2)
summary(t2)

# Add new feature/variable - family size
t2$family_size <- t2$sibsp + t2$parch + 1

# Convert data type where appropriate
t2$survived <- factor(t2$survived)
t2$sex <- factor(t2$sex)
t2$embarked <- factor(t2$embarked)
t2$pclass <- factor(t2$pclass, order=TRUE, levels = c(1,2,3))


################################
################################
# SPLIT TRAINING AND TEST SETS
################################
################################

n <- nrow(t2)

set.seed(100)
train_set <- sample(n, round(0.5*n))
test_set <- setdiff(1:n, train_set)


###################
###################
# SET UP THE TASK
###################
###################

task <- TaskClassif$new('titanic', backend=t2, target = 'survived')
task$select(c('age', 'fare', 'embarked', 'sex', 'pclass', 'family_size'))
task

msr()  #show all measures
measure <- msr('classif.ce')

# Some variables are factors for which some methods do not support
# Hence, we need to convert them to numerical values 

# Factor encoder
# method="treatment": create n-1 columns leaving out the first factor level
# method="one-hot": create a new column for each factor level
# Reference -- https://rdrr.io/cran/mlr3pipelines/man/mlr_pipeops_encode.html
fencoder <- po("encode", method="treatment",
               affect_columns=selector_type("factor"))

# Change ordered to integer
ord_to_int <- po("colapply", applicator=as.integer,
                 affect_columns=selector_type("ordered"))


# Some methods require tuning the hyperparameters, and we will later use the following:
tuner <- tnr('grid_search')
terminator <- trm('evals', n_evals = 20)


#############################################
#############################################
# EXTEND RESULTS WITH DIFFERENT CLASSIFIERS
#############################################
#############################################

# Reference:
# https://cran.r-project.org/web/packages/mlr3learners/mlr3learners.pdf


########################
# Logistic regression 1
########################

learner_lr <- lrn("classif.log_reg")

gc_lr <- po('imputemean', affect_columns=selector_type("numeric")) %>>%
  po('imputemode', affect_columns=selector_type(c("factor","ordered"))) %>>%
  po(learner_lr)

# Reference on imputation methods -- https://mlr3gallery.mlr-org.com/posts/2020-01-30-impute-missing-levels/
# 1: imputeconstant
# 2: imputehist
# 3: imputelearner
# 4: imputemean
# 5: imputemedian
# 6: imputemode
# 7: imputeoor
# 8: imputesample

glrn_lr <- GraphLearner$new(gc_lr)

glrn_lr$train(task, row_ids = train_set)
glrn_lr$predict(task, row_ids = test_set)$score()


########################
# Logistic regression 2
########################

learner_lr2 <- lrn("classif.log_reg")

gc_lr2 <- po('imputemean', affect_columns=selector_type("numeric")) %>>%
  po('imputemode', affect_columns=selector_type(c("factor","ordered"))) %>>%
  fencoder %>>% ord_to_int %>>%
  po('scale') %>>%
  po(learner_lr2)

# Reference on feature scaling -- https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/

glrn_lr2 <- GraphLearner$new(gc_lr2)

glrn_lr2$train(task, row_ids=train_set)
glrn_lr2$predict(task, row_ids=test_set)$score() 



#####################
# Gradient boosting
#####################

#set.seed(10)

learner_gb <- lrn("classif.xgboost")

gc_gb <- po('imputemean', affect_columns=selector_type("numeric")) %>>%
  po('imputemode', affect_columns=selector_type(c("factor","ordered"))) %>>%
  fencoder %>>% ord_to_int %>>%
  po(learner_gb)

glrn_gb <- GraphLearner$new(gc_gb)

glrn_gb$train(task, row_ids = train_set)
glrn_gb$predict(task, row_ids = test_set)$score() 


#################################
# Penalised logistic regression
#################################

learner_plr <- lrn('classif.glmnet')

gc_plr <- po('imputemean', affect_columns=selector_type("numeric")) %>>%
  po('imputemode', affect_columns=selector_type(c("factor","ordered"))) %>>%
  fencoder %>>% ord_to_int %>>%
  po('scale') %>>% 
  po(learner_plr)

glrn_plr <- GraphLearner$new(gc_plr)

tune_lambda <- ParamSet$new (list(
  ParamDbl$new('classif.glmnet.lambda', lower=0.001, upper=2)
))

at_plr <- AutoTuner$new(
  learner = glrn_plr,
  resampling = rsmp('cv', folds=3),
  measure = measure,
  search_space = tune_lambda,
  terminator = terminator,
  tuner = tuner
)

at_plr$train(task, row_ids = train_set)
at_plr$predict(task, row_ids = test_set)$score()


#######################
# Classification tree
#######################

learner_tree <- lrn("classif.rpart")

gc_tree <- po('imputemean', affect_columns=selector_type("numeric")) %>>%
  po('imputemode', affect_columns=selector_type(c("factor","ordered"))) %>>%
  po(learner_tree)

glrn_tree <- GraphLearner$new(gc_tree)

glrn_tree$train(task, row_ids = train_set)
glrn_tree$predict(task, row_ids = test_set)$score() 


#################
# Random forest
#################

#set.seed(10)

learner_rf <- lrn('classif.ranger') 
learner_rf$param_set$values <- list(min.node.size=4)

gc_rf <- po('imputemean', affect_columns=selector_type("numeric")) %>>%
  po('imputemode', affect_columns=selector_type(c("factor","ordered"))) %>>%
  po(learner_rf)

glrn_rf <- GraphLearner$new(gc_rf)

tune_ntrees <- ParamSet$new (list(
  ParamInt$new('classif.ranger.num.trees', lower = 50, upper = 600)
))

at_rf <- AutoTuner$new(
  learner = glrn_rf,
  resampling = rsmp('cv', folds=3),
  measure = measure,
  search_space = tune_ntrees,
  terminator = terminator,
  tuner = tuner
)

at_rf$train(task, row_ids = train_set)
at_rf$predict(task, row_ids = test_set)$score() 


#################################
# Support vector machines (SVM)
#################################

learner_svm <- lrn("classif.svm")

gc_svm <- po('imputemean', affect_columns=selector_type("numeric")) %>>%
  po('imputemode', affect_columns=selector_type(c("factor","ordered"))) %>>%
  fencoder %>>% ord_to_int %>>%
  po('scale') %>>%
  po(learner_svm)

glrn_svm <- GraphLearner$new(gc_svm)

glrn_svm$train(task, row_ids = train_set)
glrn_svm$predict(task, row_ids = test_set)$score() 


####################################
####################################
# BENCHMARKING -- COMPARE RESULTS
####################################
####################################

set.seed(1) # for reproducible results

# List of learners
lrn_list <- list(
  glrn_lr,
  glrn_gb,
  at_plr,
  glrn_tree,
  at_rf,
  glrn_svm
)

# Set the benchmark design and run the comparisons
bm_design <- benchmark_grid(task=task, resamplings=rsmp('cv', folds=3), 
                            learners=lrn_list)
bmr <- benchmark(bm_design, store_models=TRUE)

# Visualise comparisons with boxplots
library(mlr3viz)
library(ggplot2)
autoplot(bmr) + theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Print overall measure for each classification model
bmr$aggregate(measure)
