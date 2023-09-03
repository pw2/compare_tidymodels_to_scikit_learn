
#### Tidytuesday: Compare tidymodels to scikit learn

## Load packages and data
library(tidyverse)
library(tidymodels)

theme_set(theme_light())

dat <- read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2023/2023-04-04/soccer21-22.csv')
dat %>%
  head()

## EDA
dat %>%
  summarize(across(.cols = everything(),
                   ~sum(is.na(.x)))) %>%
  t()

# FTR = Full Time Result (our dependent variable)
dat %>%
  ggplot(aes(x = FTR)) +
  geom_bar() +
  labs(x = "Game Result",
       y = "Count",
       title = "Premier League Soccer Results",
       subtitle = "Home team wins more than the away team")

dat %>%
  count(FTR) %>%
  mutate(pct = scales::percent(n / sum(n), accuracy = 0.1))


# create a data set with some features to try and predict game outcome
dat_model <- dat %>%
  select(FTR, HTHG, HTAG, HS:AR)

dat_model %>%
  head()

dat_model %>%
  pivot_longer(cols = -FTR) %>%
  ggplot(aes(x = FTR, y = value)) +
  geom_boxplot() +
  facet_wrap(~name, scales = "free") +
  theme(strip.background = element_rect(fill = "black"),
        strip.text = element_text(face = "bold")) +
  labs(x = "Game Result",
       y = NULL,
       title = "Potential Predictors of Premier League Game Result",
       subtitle = "Note the y-axis is specific to each facet variable")

## Train/Test Split
set.seed(1056)
dat_split <- initial_split(dat_model, strata = FTR)

train <- training(dat_split)
testing <- testing(dat_split)

train %>%
  count(FTR) %>%
  mutate(pct = n / sum(n))

testing %>%
  count(FTR) %>%
  mutate(pct = n / sum(n))

## Model Engine
rf <- rand_forest(mtry = tune(), trees = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("classification")

## Cross Validation Split
set.seed(468)
cv_folds <- vfold_cv(
  data = train,
  v = 5
)

cv_folds

## Model Recipe
rec <- recipe(FTR ~ ., data = train)

rec %>%
  prep()

rec %>%
  prep() %>%
  bake(new_data = NULL)


## Workflow
wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf)

wf

## Hyperparameter tuning
tune_grid <- grid_regular(
  mtry(range = c(1, 14)),
  trees(range = c(400, 900)),
  levels = 5
)


ctrl <- control_resamples(save_pred = TRUE)

doParallel::registerDoParallel(cores = 5)

rf_tune <- tune_grid(
  wf,
  resamples = cv_folds,
  grid =tune_grid,
  control = ctrl
)

doParallel::stopImplicitCluster()

## View the model performance & extract the best model
# plot results
rf_tune %>%
  autoplot() +
  theme_minimal() +
  labs(title='Hyperparameter Tuning Results')

# collect metrics
rf_tune %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  arrange(desc(mean))

# get best model
best_mtry <- rf_tune %>%
  select_best("roc_auc") %>%
  pull(mtry)

best_trees <- rf_tune %>%
  select_best("roc_auc") %>%
  pull(trees)

best_mtry
best_trees

## Finalize workflow
wf_final <- finalize_workflow(wf, select_best(rf_tune, "roc_auc"))

## Fit final model
rf_fit_final <- wf_final %>% 
  last_fit(
    split = dat_split
  )

rf_fit_final

## plot the variables of importance
library(vip)

rf_fit_final %>%
  extract_fit_parsnip() %>% 
  vip(geom = "col",
      aesthetics = list(
        color = "black",
        fill = "palegreen",
        alpha = 0.5)) +
  theme_classic()

## Get predictions on test data
fit_test <- rf_fit_final %>% 
  collect_predictions()

fit_test

table(
  actual = fit_test$FTR,
  predicted = fit_test$.pred_class
)

## Evaluate model performance
# Collect metrics
collect_metrics(rf_fit_final)

## plot of prediction vs true outcome

fit_test %>%
  roc_curve(
    truth = FTR,
    .pred_A, .pred_D, .pred_H
  ) %>% 
  autoplot()


## Save Model & Make Predictions on New Data
rf_model <- wf_final %>% 
  fit(dat_model) %>%
  extract_fit_parsnip()

rf_model

# Save model
save(rf_model, file = "rf_model.rda")

# load model
load("rf_model.rda")

# create some "new" data
new_dat <- dat_model %>% slice(c(181, 243))
new_dat

new_dat %>%
  mutate(predict(rf_model, new_data = new_dat))
