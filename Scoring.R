###                   Khanh TRUONG - Thanh-Tam NGUYEN
###                           Scoring Project

### Import data
### ----------------------------------------------------------------------------

library(dplyr)
library(sas7bdat)
library(ggplot2)
library(gridExtra)
library(Hmisc)
library(reshape2)
library(randomForest) # random forest
library(xgboost) # xgboost
library(missForest) # for impute missing value


theme_update(plot.title = element_text(hjust = 0.5)) # for ploting title

# import data
fund <- read.sas7bdat('cours_d1.sas7bdat')

# dependent variable
fund[fund$Y==9, 'Y' ] <- 0 # change Y into 0 and 1

# factor variable
factor_cols <- c('DETVIE', 'DETIMMO', 'DETCONSO', 'DETREV', 'DETLIQ', 'PEA',
                 'DETBLO', 'CPTTIT', 'PSOC')

# continuous variable
cont_cols <- names(fund)[!names(fund) %in% c(factor_cols, 'Y', 'NUPER')]
# Y is dependent variable
# NUPER is identity


### ----------------------------------------------------------------------------
### Discretize continuous varibles

# example of AGE
ggplot(data=fund, aes(AGE)) +
  geom_density(fill='#D3D3D3') +
  geom_vline(xintercept=quantile(fund$AGE), size=1.5, color='darkblue') +
  labs(title='AGE Density') +
  geom_text(aes(x=24, y=0.01), label="1", size=5) +
  geom_text(aes(x=36, y=0.01), label="2", size=5) +
  geom_text(aes(x=47, y=0.01), label="3", size=5) +
  geom_text(aes(x=75, y=0.01), label="4", size=5)


cont_cols1 <- c('NBJDE', 'SLDPEA', 'SLDBLO') # can't discretize those variables
# into 4 quarters because e.g. min = median
summary(fund[cont_cols1])

# Percentage of zero
nrow(fund[fund['NBJDE']==0 & is.na(fund$NBCRE)==FALSE,])/
  nrow(fund[is.na(fund$NBCRE)==FALSE,])

nrow(fund[fund['SLDPEA']==0 & is.na(fund$SLDPEA)==FALSE,])/
  nrow(fund[is.na(fund$SLDPEA)==FALSE,])

nrow(fund[fund['SLDBLO']==0 & is.na(fund$SLDBLO)==FALSE,])/
  nrow(fund[is.na(fund$SLDBLO)==FALSE,])

cont_cols2 <- cont_cols[!cont_cols %in% cont_cols1]
# Discretize continuous variables in cont_cols2
for (i in cont_cols2){
  name <- paste0(i,'_group')
  
  fund[name] <- as.numeric(cut(fund[,i],
                                breaks = quantile(fund[,i],
                                                  probs = seq(0, 1, 0.25),
                                                  na.rm = TRUE),
                                right = FALSE))
  
  fund[is.na(fund[i])==FALSE & fund[i]==max(fund[i], na.rm = TRUE), name] <-
    max(fund[name], na.rm = TRUE)
}
rm(name, i)


# Discretize continuous variables in cont_cols1
fund['NBJDE_group'] <-  fund['NBJDE']
fund[fund[,'NBJDE_group']!=0
      & is.na(fund[,'NBJDE_group'])==FALSE, 'NBJDE_group'] <-
  as.numeric(cut(fund[fund[,'NBJDE_group']!=0
                       & is.na(fund[,'NBJDE_group'])==FALSE, 'NBJDE_group'],
                 breaks=quantile(fund[fund[,'NBJDE_group']!=0, 'NBJDE_group'],
                                 probs = seq(0, 1, 0.25),
                                 na.rm = TRUE),
                 right=FALSE))
fund[fund$NBJDE==max(fund$NBJDE, na.rm=TRUE) & is.na(fund$NBJDE)==FALSE,
      'NBJDE_group'] <-
  max(fund['NBJDE_group'], na.rm = TRUE)


fund['SLDPEA_group'] <-  fund['SLDPEA']
fund[fund[,'SLDPEA_group']!=0
      & is.na(fund[,'SLDPEA_group'])==FALSE, 'SLDPEA_group'] <-
  as.numeric(cut(fund[fund[,'SLDPEA_group']!=0
                       & is.na(fund[,'SLDPEA_group'])==FALSE, 'SLDPEA_group'],
                 breaks=quantile(fund[fund[,'SLDPEA_group']!=0,'SLDPEA_group'],
                                 probs = seq(0, 1, 0.25),
                                 na.rm = TRUE),
                 right = FALSE))
fund[fund$SLDPEA==max(fund$SLDPEA, na.rm=TRUE) & is.na(fund$SLDPEA)==FALSE,
      'SLDPEA_group'] <-
  max(fund['SLDPEA_group'], na.rm = TRUE)


fund['SLDBLO_group'] <- fund['SLDBLO']
fund[fund[,'SLDBLO_group']!=0
      & is.na(fund[,'SLDBLO_group'])==FALSE, 'SLDBLO_group'] <-
  as.numeric(cut(fund[fund[,'SLDBLO_group']!=0
                       & is.na(fund[,'SLDBLO_group'])==FALSE, 'SLDBLO_group'],
                 breaks=quantile(fund[fund[,'SLDBLO_group']!=0,'SLDBLO_group'],
                                 probs = seq(0, 1, 0.25),
                                 na.rm = TRUE),
                 right = FALSE))
fund[fund$SLDBLO==max(fund$SLDBLO, na.rm=TRUE) & is.na(fund$SLDBLO)==FALSE,
      'SLDBLO_group'] <-
  max(fund['SLDBLO_group'], na.rm = TRUE)



### ----------------------------------------------------------------------------
### Convert the discretized variables into factors (several levels)

cont_group_cols <- paste0(cont_cols, '_group')
fund[is.na(fund)] <- 'NA' # create NA group
fund[c(factor_cols, cont_group_cols)] <-
  lapply(fund[c(factor_cols, cont_group_cols)], factor)


### ----------------------------------------------------------------------------
### Chisquare

model <- paste(unlist(c(cont_group_cols, factor_cols)), collapse=' + ')
model <- paste('Y ~', model)

chisquare_df <- chiSquare(model, data=fund) # chisquare
chisquare_df <- as.data.frame.matrix(chisquare_df) # transform to dataframe
chisquare_df['Variable'] <- rownames(chisquare_df) # add column 'Variable'
chisquare_df['Variable']<-gsub('_group', '', chisquare_df$Variable)# remove _group
chisquare_df <- chisquare_df[order(-chisquare_df$chisquare),] # sort data
chisquare_df$Variable <- as.character(chisquare_df$Variable)#transform character
chisquare_df$Variable <- factor(chisquare_df$Variable,
                                levels=unique(chisquare_df$Variable)) # to factor
rownames(chisquare_df) <- NULL # remove rownames
colnames(chisquare_df)[which(names(chisquare_df) == "chisquare")] <- "Chisquare"


chisquare_df['Type'] <-
  lapply(chisquare_df['Variable'],
         FUN = {function(x) ifelse(x %in% cont_cols, 'Continuous', 'Factor')})

ggplot(data=chisquare_df, aes(x=Variable, y=Chisquare)) +
  geom_point(aes(size=Chisquare, color=Type)) +
  geom_line(group=1) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title='Chisquare of Variables', y='Chisquare Value')
rm(model, chisquare_df)

### ----------------------------------------------------------------------------
### Focus on some important continuous variable

# ==============================================================================
# SLDPEA

# SLDPEA group quantile
SLDPEA_group_df <- fund %>%
  group_by(SLDPEA_group) %>%
  summarise(mean_y = mean(Y), count_y = n())

g_SLDPEA <- ggplot(SLDPEA_group_df, aes(SLDPEA_group, mean_y)) +
  geom_line(color='blue', group=1) +
  scale_size(range = c(2,5)) + 
  geom_point(aes(size=count_y), color='darkblue') +
  labs(x = NULL,
       y='% Subscribe',
       title='SLDPEA Group and Subscribe Rate',
       size='# Customer') +
  geom_text(aes(label=count_y),hjust=-0.3, vjust=0.1)

g_SLDPEA
rm(SLDPEA_group_df)


# group 1,2,3,4 together
fund['SLDPEA_group2'] <- fund['SLDPEA_group']
fund[fund$SLDPEA_group2!=0 & fund$SLDPEA_group2 != 'NA','SLDPEA_group2'] <- 1
fund$SLDPEA_group2 <- as.character(fund$SLDPEA_group2)
fund$SLDPEA_group2 <- as.factor(fund$SLDPEA_group2)
summary(fund$SLDPEA_group2)


SLDPEA_group2_df <- fund %>%
  group_by(SLDPEA_group2) %>%
  summarise(mean_y = mean(Y), count_y = n())
SLDPEA_group2_df <- SLDPEA_group2_df[complete.cases(SLDPEA_group2_df), ]

g_SLDPEA2 <- ggplot(SLDPEA_group2_df, aes(SLDPEA_group2, mean_y)) +
  geom_line(color='blue', group=1) +
  scale_size(range = c(2,5)) + 
  geom_point(aes(size=count_y), color='darkblue') +
  labs(x = 'SLDPEA Group',
       y='% Subscribe',
       # title='SLDPEA Group 2 and Subscribe Rate',
       size='# Customer') +
  geom_text(aes(label=count_y),hjust=-0.5, vjust=0.1) +
  theme(legend.position="none")

g_SLDPEA2
rm(SLDPEA_group2_df)

grid.arrange(g_SLDPEA, g_SLDPEA2)
rm(g_SLDPEA, g_SLDPEA2)

# ==============================================================================
# SURFIN

# SURFIN group quantile
SURFIN_group_df <- fund %>%
  group_by(SURFIN_group) %>%
  summarise(mean_y = mean(Y), count_y = n())

g_SURFIN <- ggplot(SURFIN_group_df, aes(SURFIN_group, mean_y)) +
  geom_line(color='blue', group=1) +
  scale_size(range = c(4.5,5)) + 
  geom_point(aes(size=count_y), color='darkblue') +
  labs(x = NULL,
       y='% Subscribe',
       title='SURFIN Group and Subscribe Rate',
       size='# Customer') +
  geom_text(aes(label=count_y),hjust=-0.3, vjust=0.7)

g_SURFIN
rm(SURFIN_group_df)

# Split into more groups
fund['SURFIN_group2'] <-
  as.numeric(cut(fund[,'SURFIN'],
                 breaks = quantile(fund[,'SURFIN'],
                                   probs = c(0,0.3,0.4,0.5,0.6,0.9,1.0),
                                   na.rm = TRUE),
                 right = FALSE))
fund[is.na(fund$SURFIN_group2) == TRUE, 'SURFIN_group2'] <-
  max(fund['SURFIN_group2'], na.rm = TRUE)
fund['SURFIN_group2'] <- factor(fund[['SURFIN_group2']])

SURFIN_group2_df <- fund %>%
  group_by(SURFIN_group2) %>%
  summarise(mean_y = mean(Y), count_y = n())

g_SURFIN2 <- ggplot(SURFIN_group2_df, aes(SURFIN_group2, mean_y)) +
  geom_line(color='blue', group=1) +
  scale_size(range = c(3,5)) + 
  geom_point(aes(size=count_y), color='darkblue') +
  labs(x = 'SURFIN Group',
       y='% Subscribe',
       # title='SURFIN Group and Subscribe Rate',
       size='# Customer') +
  geom_text(aes(label=count_y),hjust=-0.2, vjust=0.5) +
  #scale_x_continuous("SURFIN group",
  #                   labels = as.character(SURFIN_group2_df$SURFIN_group2),
  #                   breaks = SURFIN_group2_df$SURFIN_group2) +
  theme(legend.position="none")

g_SURFIN2
rm(SURFIN_group2_df)

grid.arrange(g_SURFIN, g_SURFIN2)
rm(g_SURFIN, g_SURFIN2)

# ==============================================================================
# SLDBLO
# SLDBLO group quantile
SLDBLO_group_df <- fund %>%
  group_by(SLDBLO_group) %>%
  summarise(mean_y = mean(Y), count_y = n())

g_SLDBLO <- ggplot(SLDBLO_group_df, aes(SLDBLO_group, mean_y)) +
  geom_line(color='blue', group=1) +
  scale_size(range = c(3,5)) + 
  geom_point(aes(size=count_y), color='darkblue') +
  labs(x = NULL,
       y='% Subscribe',
       title='SLDBLO Group and Subscribe Rate',
       size='# Customer') +
  geom_text(aes(label=count_y),hjust=-0.3, vjust=0.7)

# group 0 and 2 together
# group 1 and 3 together

fund['SLDBLO_group2'] <- NA
fund[fund$SLDBLO_group==0 | fund$SLDBLO_group==2, 'SLDBLO_group2'] <- 1
fund[fund$SLDBLO_group==1 | fund$SLDBLO_group==3, 'SLDBLO_group2'] <- 2
fund[fund$SLDBLO_group==4, 'SLDBLO_group2'] <- 3
fund[fund$SLDBLO_group=='NA', 'SLDBLO_group2'] <- 'NA'
fund$SLDBLO_group2 <- factor(as.character(fund$SLDBLO_group2))

SLDBLO_group2_df <- fund %>%
  group_by(SLDBLO_group2) %>%
  summarise(mean_y = mean(Y), count_y = n())

g_SLDBLO2 <- ggplot(SLDBLO_group2_df, aes(SLDBLO_group2, mean_y)) +
  geom_line(color='blue', group=1) +
  scale_size(range = c(3,5)) + 
  geom_point(aes(size=count_y), color='darkblue') +
  labs(x = 'SLDBLO Group',
       y='% Subscribe',
       # title='SLDBLO Group and Subscribe Rate',
       size='# Customer') +
  geom_text(aes(label=count_y),hjust=-0.3, vjust=0.5) +
  theme(legend.position="none")
g_SLDBLO2


grid.arrange(g_SLDBLO, g_SLDBLO2)
rm(SLDBLO_group_df, g_SLDBLO, SLDBLO_group2_df, g_SLDBLO2)


# ==============================================================================
# AGE

# Age group quantile
AGE_group_df <- fund %>%
  group_by(AGE_group) %>%
  summarise(mean_y = mean(Y), count_y = n())

g_age <- ggplot(AGE_group_df, aes(AGE_group, mean_y)) +
  geom_line(size=1, color='blue', group=1) +
  scale_size(range = c(4.5,5)) + 
  geom_point(aes(size=count_y), color='darkblue') +
  labs(x = NULL,
       y='% Subscribe',
       title='Age Quarters and Subscribe Rate',
       size='# Customer')

g_age
rm(AGE_group_df)

# Age continuous
summary(fund$AGE)
fund$AGE_group2 <- cut(fund$AGE,
                       breaks=c(19, 30, 40, 50, 60, 70, 80, 90, 100),
                       right = FALSE)
fund[fund$AGE==max(fund$AGE, na.rm=TRUE), 'AGE_group2'] <- '[90,100)'

AGE_group2_df <- fund %>%
  group_by(AGE_group2) %>%
  summarise(mean_y = mean(Y), count_y = n())

g_age2 <- ggplot(AGE_group2_df, aes(AGE_group2, mean_y)) +
  geom_line(size=1, color='blue', group = 1) +
  scale_size(range = c(3,5)) + 
  geom_point(aes(size=count_y), color='darkblue') +
  labs(x = NULL,
       y='% Subscribe',
       title='Age and Subscribe Rate',
       size='# Customer') +
  theme(legend.position="none")
g_age2
rm(AGE_group2_df)

# consider replace AGE by |45-AGE|
fund$AGE_group3 <- abs(45 - fund$AGE)
summary(fund$AGE_group3)

fund$AGE_group3 <- cut(fund$AGE_group3,
                        breaks=c(0, 10, 20, 30, 40, 55),
                        right=FALSE)
fund[fund$AGE==max(fund$AGE, na.rm=TRUE), 'AGE_group3'] <- '[40,55)'

AGE_group3_df <- fund %>%
  group_by(AGE_group3) %>%
  summarise(mean_y = mean(Y), count_y = n())

g_age3 <- ggplot(AGE_group3_df, aes(AGE_group3, mean_y, group = 1)) +
  geom_line(size=1, color='blue') +
  scale_size(range = c(3,5)) + 
  geom_point(aes(size=count_y), color='darkblue') +
  labs(x = NULL,
       y='% Subscribe',
       title='|45-Age| and Subscribe Rate',
       size='# Customer') +
  theme(legend.position="none")

g_age3
rm(AGE_group3_df)

grid.arrange(g_age, g_age2, g_age3)
rm(g_age, g_age2, g_age3)


fund$AGE_group3 <-  as.numeric(fund$AGE_group3)
fund['AGE_group3'] <- factor(fund[['AGE_group3']])



# Update SLDPEA_group, SURFIN_group and AGE_group
fund['SLDPEA_group'] <- fund['SLDPEA_group2']
fund['SURFIN_group'] <- fund['SURFIN_group2']
fund['AGE_group'] <- fund['AGE_group3']


### Split into train and test sets
### ----------------------------------------------------------------------------

fund_pos <- fund[fund$Y == 1,] # positive response set
fund_neg <- fund[fund$Y == 0,] # negative response set

train_size <- 0.7 # training set size

# Take postive reponse for training set
set.seed(42)
train_pos_index <- sample(1:nrow(fund_pos),
                          round(train_size*nrow(fund_pos)),
                          replace = FALSE)
train_pos <- fund_pos[train_pos_index, ]

# Take negative reponse for training set
set.seed(42)
train_neg_index <- sample(1:nrow(fund_neg),
                          round(train_size*nrow(fund_neg)),
                          replace = FALSE)
train_neg <- fund_neg[train_neg_index, ]


train <- rbind(train_pos, train_neg) # Merge to get train set
test <- anti_join(fund, train) # Exclude from full data to get test set
rm(fund_neg, fund_pos, train_neg, train_pos,
   train_neg_index, train_pos_index, train_size)


# Update AGE_group3 SLDPEA_group2 SURFIN_group2 in cont_group_cols
train_logit <- train[c(cont_group_cols, factor_cols, 'Y')]
test_logit <- test[c(cont_group_cols, factor_cols, 'Y')]


### ----------------------------------------------------------------------------
### Data for random forest and xgboost


# import again the original data
fund <- read.sas7bdat('cours_d1.sas7bdat')
fund[fund$Y==9, 'Y' ] <- 0 # change Y into 0 and 1
fund$NUPER <- NULL
fund[c(factor_cols,'Y')] <- lapply(fund[c(factor_cols,'Y')], factor)

fund_imputed <- missForest(fund)
fund_imputed <- fund_imputed$ximp

# split train and test
fund_imputed_pos <- fund_imputed[fund_imputed$Y == 1,] # positive response set
fund_imputed_neg <- fund_imputed[fund_imputed$Y == 0,] # negative response set

train_size <- 0.7 # training set size

# Take postive reponse for training set
set.seed(42)
train_pos_index <- sample(1:nrow(fund_imputed_pos),
                          round(train_size*nrow(fund_imputed_pos)),
                          replace = FALSE)
train_pos <- fund_imputed_pos[train_pos_index, ]

# Take negative reponse for training set
set.seed(42)
train_neg_index <- sample(1:nrow(fund_imputed_neg),
                          round(train_size*nrow(fund_imputed_neg)),
                          replace = FALSE)
train_neg <- fund_imputed_neg[train_neg_index, ]


train <- rbind(train_pos, train_neg) # Merge to get train set
test <- anti_join(fund_imputed, train) # Exclude from full data to get test set
rm(fund_imputed_neg, fund_imputed_pos, train_neg, train_pos,
   train_neg_index, train_pos_index, train_size)


### ----------------------------------------------------------------------------
### Logistic Regression

# base intercept only model
base_mod <- glm(Y~1, data = train_logit, family = "binomial")

# full model with all predictors
all_mod <- glm(Y~., data = train_logit, family = "binomial") 

# perform step-wise algorithm
logit <- step(base_mod, scope = list(lower = base_mod, upper = all_mod),
                 direction = "both", trace = 0, steps = 1000)
rm(base_mod, all_mod)
summary(logit)

# get the shortlisted variable
var_final <- names(unlist(logit[[1]]))

# remove intercept
var_final <- var_final[!var_final %in% "(Intercept)"]
var_final
rm(var_final)


train_predict <- data.frame(predict(logit, train_logit, type = 'response'),
                            train_logit$Y)
                           
names(train_predict) <- c('prob_logit', 'actual_label')
train_predict$actual_label <- factor(train_predict$actual_label)


g_logit <- ggplot(data=train_predict, aes(y=prob_logit, x=actual_label, color=actual_label)) +
  geom_boxplot() +
  labs(x='Actual Subsribe',
       y='Fitted Probability',
       title='Logistic Regression') +
  theme(legend.position="none")
g_logit


### ----------------------------------------------------------------------------
### Random Forest

rf <- randomForest(Y~., data=train)
train_predict['prob_rf'] <- predict(rf, train, type="prob")[,2]

g_rf <- ggplot(data=train_predict,
               aes(y=prob_rf, x=actual_label, color=actual_label)) +
  geom_boxplot() +
  labs(x='Actual Subscribe',
       y=NULL,
       title='Random Forest') +
  theme(legend.position="none")
g_rf


### ----------------------------------------------------------------------------
### xgboosting

params_xgb = expand.grid(
  objective = "binary:logistic",
  eta = 0.15,
  max_depth = 5,
  colsample_bytree = 0.8,
  min_child_weight = 3,
  subsample = 1
)

xgb <- xgboost(data = data.matrix(subset(train, select=-c(Y))),
               label = as.numeric(train$Y)-1,
               params = params_xgb,
               missing = NA,
               nrounds = 500)
rm(params_xgb)

train_predict['prob_xgb'] <-
  predict(xgb, data.matrix(subset(train, select=-c(Y))))

g_xgb <- ggplot(data=train_predict,
               aes(y=prob_xgb, x=actual_label, color=actual_label)) +
  geom_boxplot() +
  labs(x='Actual Subscribe',
       y=NULL,
       title='XGBoost') +
  theme(legend.position="none")
g_xgb

grid.arrange(g_logit, g_rf, g_xgb,
             layout_matrix = rbind(c(1,2,3),c(1,2,3)))
rm(g_logit, g_rf, g_xgb)


### ----------------------------------------------------------------------------
### Predict on testing set

prob_logit <- predict(logit, test_logit, type = 'response')
prob_rf <- predict(rf, test, type="prob")[,2]
prob_xgb <- predict(xgb, data.matrix(subset(test, select=-c(Y))))
actual_label <- test$Y

test_predict <- data.frame(prob_logit,
                           prob_rf,
                           prob_xgb,
                           actual_label)

rm(prob_logit, prob_rf, prob_xgb, actual_label)

### ----------------------------------------------------------------------------
### Model evaluation


# ROC Curve
sens_logit <- c()
spec_logit <- c()

sens_rf <- c()
spec_rf <- c()

sens_xgb <- c()
spec_xgb <- c()

for (i in seq(0, 1, 0.05)){
  
  # Sensitivity and specificity of logit
  test_predict['label_logit'] <- as.numeric(test_predict['prob_logit'] >= i)
  
  sens_logit_i <- nrow(test_predict[test_predict$actual_label==1
                                      & test_predict$label_logit==1, ]) /
    nrow(test_predict[test_predict$actual_label==1,])
  
  spec_logit_i <- nrow(test[test_predict$actual_label==0
                              & test_predict$label_logit==0, ]) /
    nrow(test_predict[test_predict$actual_label==0,])
  
  sens_logit <- append(sens_logit, sens_logit_i)
  spec_logit <- append(spec_logit, spec_logit_i)
  
  test_predict['label_logit'] <- NULL
  
  
  # Sensitivity and specificity of random forest
  
  test_predict['label_rf'] <- as.numeric(test_predict['prob_rf'] >= i)
  
  sens_rf_i <- nrow(test_predict[test_predict$actual_label==1
                                     & test_predict$label_rf==1, ]) /
    nrow(test_predict[test_predict$actual_label==1,])
  
  spec_rf_i <- nrow(test[test_predict$actual_label==0
                             & test_predict$label_rf==0, ]) /
    nrow(test_predict[test_predict$actual_label==0,])
  
  sens_rf <- append(sens_rf, sens_rf_i)
  spec_rf <- append(spec_rf, spec_rf_i)
  
  test_predict['label_rf'] <- NULL
  
  # Sensitivity and specificity of xgb
  test_predict['label_xgb'] <- as.numeric(test_predict['prob_xgb'] >= i)
  
  sens_xgb_i <- nrow(test_predict[test_predict$actual_label==1
                                     & test_predict$label_xgb==1, ]) /
    nrow(test_predict[test_predict$actual_label==1,])
  
  spec_xgb_i <- nrow(test[test_predict$actual_label==0
                             & test_predict$label_xgb==0, ]) /
    nrow(test_predict[test_predict$actual_label==0,])
  
  sens_xgb <- append(sens_xgb, sens_xgb_i)
  spec_xgb <- append(spec_xgb, spec_xgb_i)
  
  test_predict['label_xgb'] <- NULL
  
  
}
rm(i, sens_logit_i, spec_logit_i, sens_rf_i, spec_rf_i, sens_xgb_i, spec_xgb_i)

roc <- data.frame(sens_logit, spec_logit, sens_rf, spec_rf, sens_xgb, spec_xgb)
rm(sens_logit, spec_logit, sens_rf, spec_rf, sens_xgb, spec_xgb)
roc_curve <- ggplot(data=roc) +
  geom_line(aes(x=1-spec_logit, y=sens_logit), size=1.2, color='darkblue') +
  geom_line(aes(x=1-spec_rf, y=sens_rf), size=1.2, color='darkgreen') +
  geom_line(aes(x=1-spec_xgb, y=sens_xgb), size=1.2, color='darkred') +
  geom_abline(intercept=0 , slope=1) +
  labs(title='ROC Curve', x='1-Specificity', y='Sensitivity')
roc_curve


# Concentration Curve
percent_client <- seq(0, 1, 0.05)
percent_subcribe_logit <- c()
percent_subcribe_rf <- c()
percent_subcribe_xgb <- c()

for (i in percent_client){
  
  # Concentration of logit model
  test_predict <- test_predict[order(-test_predict$prob_logit),]
  
  percent_subcribe_logit_i <-
    sum(as.numeric(test_predict[1:round(i*nrow(test_predict),0),'actual_label'])-1) /
    nrow(test_predict[test_predict$actual_label==1,])
  
  percent_subcribe_logit <- append(percent_subcribe_logit,
                                   percent_subcribe_logit_i)
  
  
  # Concentration of random forest model
  test_predict <- test_predict[order(-test_predict$prob_rf),]
  
  percent_subcribe_rf_i <-
    sum(as.numeric(test_predict[1:round(i*nrow(test_predict),0),'actual_label'])-1) /
    nrow(test_predict[test_predict$actual_label==1,])
  
  percent_subcribe_rf <- append(percent_subcribe_rf,
                                percent_subcribe_rf_i)
  
  # Concentration of random forest model
  test_predict <- test_predict[order(-test_predict$prob_xgb),]
  
  percent_subcribe_xgb_i <-
    sum(as.numeric(test_predict[1:round(i*nrow(test_predict),0),'actual_label'])-1) /
    nrow(test_predict[test_predict$actual_label==1,])
  
  percent_subcribe_xgb <- append(percent_subcribe_xgb,
                                percent_subcribe_xgb_i)
           
}
rm(i, percent_subcribe_logit_i, percent_subcribe_rf_i, percent_subcribe_xgb_i)

concentration <- data.frame(percent_client,
                            percent_subcribe_logit,
                            percent_subcribe_rf,
                            percent_subcribe_xgb)

rm(percent_client,
   percent_subcribe_logit,
   percent_subcribe_rf,
   percent_subcribe_xgb)


concentration_curve <- ggplot(data=concentration) +
  geom_line(aes(x=percent_client, y=percent_subcribe_logit, color='Logit'),
            size=1.2) +
  geom_line(aes(x=percent_client, y=percent_subcribe_rf, color='Random Forest'),
            size=1.2) +
  geom_line(aes(x=percent_client, y=percent_subcribe_xgb, color='XGBoost'),
            size=1.2) +

  scale_color_manual('Model',
                     values = c(
                       'Logit' = 'darkblue',
                       'Random Forest' = 'darkgreen',
                       'XGBoost' = 'darkred')) +
  geom_abline(intercept=0 , slope=1) +
  
  labs(x='% Clients',
       y='% Subscribe',
       title='Concentration Curve',
       fill='Model') +
  theme(legend.justification = c(1, 0), legend.position = c(1, 0),
        legend.text=element_text(size=rel(0.6)),
        legend.title=element_text(size=rel(0.7)))

concentration_curve

grid.arrange(roc_curve, concentration_curve,
             layout_matrix = rbind(c(1,2),c(1,2)))


### ----------------------------------------------------------------------------
### Choose model and threshold

closeness <- sqrt((1 - percent_subcribe)**2 + percent_client**2)
concentration <- data.frame(percent_client,
                            percent_subcribe,
                            closeness)
(optimal <- concentration[concentration$closeness==min(concentration$closeness),])


geom_point(aes(x=c(0,as.numeric(optimal[1])),
               y=c(1,as.numeric(optimal[2]))),
           color='black',
           size=3) +
  geom_line(aes(x=c(0,as.numeric(optimal[1])),
                y=c(1,as.numeric(optimal[2]))),
            color='black',
            size=1) +
  
  geom_text(aes(x=0.30, y=0.92), label="Mininum Distance", size=3) +
  geom_text(aes(x=0.53, y=0.68),
            label="% Client: 36.5%\n% Subscribe: 73.2%",
            size=3)

### ----------------------------------------------------------------------------