###############################################################################
#                   APPROXIMATION DU PRIX DES VOITURES                         #
###############################################################################

# 1. Chargement des données et packages
setwd("~/dimitri/modele_lineaire")
data = read.csv("voitures_prix.csv")

set.seed(1)

library(ggplot2)
library(dplyr)
library(randomForest)
library(randomForestExplainer)
library(e1071)
library(xgboost)

# 2. Exploration et nettoyage initial
names(data)
summary(data)
length(data[, 1])

anyDuplicated(data) # Vérification de doublons
sum(is.na(data))    # Vérification des valeurs manquantes

levels(as.factor(data[["Model.Name"]]))

# Suppression de colonnes jugées non pertinentes
# - Model.Name (trop de catégories)
# - X (simple indice)
df = subset(data, select = -c(Model.Name, X))

head(df)

# 3. Transformation et regroupement des variables

## 3.1 Variable Color
df$Color = as.factor(df$Color)
summary(df$Color)

ggp = ggplot(df, aes(x = Color, y = log(Price))) + 
  geom_boxplot(fill = "skyblue") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggp

tapply(df$Price, df$Color, mean)

expensive_car_color = c("Black", "Assembly", "White", "Bronze", "Brown", "Burgundy", "Unlisted")
df$Color = ifelse(df$Color %in% expensive_car_color, "expensive_color", "cheap_color")
df$Color = as.factor(df$Color)

summary(df$Color)
tapply(df$Price, df$Color, mean)

## 3.2 Variable Location
df$Location = as.factor(df$Location)
tapply(df$Price, df$Location, summary)

df$Location = ifelse(df$Location %in% c("Balochistan", "Kashmir", "KPK"), 
                     "KPK_Kash_Bal", 
                     df$Location)

# Correction manuelle (cas d’étiquettes perdues, exemple)
df$Location[df$Location == 2] = "Islamabad"
df$Location[df$Location == 5] = "Punjab"
df$Location[df$Location == 6] = "Sindh"

df$Location = as.factor(df$Location)
summary(df$Location)

## 3.3 Variable Company.Name
df$Company.Name = as.factor(df$Company.Name)
summary(df$Company.Name)
tapply(df$Price, df$Company.Name, mean)

ggp = ggplot(df, aes(x = Company.Name, y = log(Price))) + 
  geom_boxplot(fill = "skyblue") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggp

expensive_brand = c("Audi", "BMW", "Hummer", "Jaguar", "Jeep", 
                    "Land", "Lexus", "Mercedes", "MINI", 
                    "Porsche", "Range", "Toyota")
medium_brand = c("Daihatsu", "DFSK", "FAW", "Honda", "KIA", 
                 "Nissan", "SsangYong", "Mazda", 
                 "Mitsubishi", "Subaru", "Volvo")

df$Company.Name = as.character(df$Company.Name)
df$Company.Name = ifelse(df$Company.Name %in% expensive_brand, 
                         "expensive_brand", df$Company.Name)
df$Company.Name = ifelse(df$Company.Name %in% medium_brand, 
                         "medium_brand", df$Company.Name)
df$Company.Name = ifelse(!(df$Company.Name %in% c("medium_brand","expensive_brand")), 
                         "cheap_brand", 
                         df$Company.Name)
df$Company.Name = as.factor(df$Company.Name)

summary(df$Company.Name)

## 3.4 Variable Engine.Type
df$Engine.Type = as.factor(df$Engine.Type)
summary(df$Engine.Type)

## 3.5 Variable Assembly
df$Assembly = as.factor(df$Assembly)
summary(df$Assembly)

## 3.6 Variable Body.Type
df$Body.Type = as.factor(df$Body.Type)
summary(df$Body.Type)

ggp = ggplot(df, aes(x = Body.Type, y = log(Price))) + 
  geom_boxplot(fill = "skyblue") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggp

Van_Hatchback = c("Van", "Mini Van", "Hatchback")
SUV_Crossover = c("SUV", "Cross Over")
df$Body.Type = as.character(df$Body.Type)
df$Body.Type = ifelse(df$Body.Type %in% Van_Hatchback, "Van_Hatchback", df$Body.Type)
df$Body.Type = ifelse(df$Body.Type %in% SUV_Crossover, "SUV_Crossover", df$Body.Type)
df$Body.Type = as.factor(df$Body.Type)
summary(df$Body.Type)

## 3.7 Variable Registration.Status
df$Registration.Status = as.factor(df$Registration.Status)
summary(df$Registration.Status)

## 3.8 Variable Transmission.Type
df$Transmission.Type = as.factor(df$Transmission.Type)
summary(df$Transmission.Type)

# 4. Analyse de la distribution du prix
ggp = ggplot(df, aes(Price))+
  geom_histogram(bins = 150, fill = "blue", color = "black", alpha = 0.7)+
  labs(title = "Distribution des prix", y = "Nombre", x = "Prix (inférieurs à 15 000 000)")+
  theme_light()+
  xlim(c(0, 15000000))
ggp

ggp = ggplot(df, aes(log(Price)))+
  geom_histogram(bins = 100, fill = "blue", color = "black", alpha = 0.7)+
  labs(title = "Distribution des prix au log", y = "Nombre", x = "Prix (au log)")+
  theme_light()
ggp

# 5. Variable Mileage (discrétisation)
summary(df$Mileage)
ggp = ggplot(df, aes(Mileage))+
  geom_histogram(bins = 200, fill = "blue", color = "black", alpha = 0.7)+
  theme_light()
ggp

cor(df$Mileage, log(df$Price))

ggp = ggplot(df, aes(x = Mileage, y = log(Price))) +
  geom_bin2d(bins = 50)+
  geom_smooth(method = "lm", color = "blue")
ggp

breaks = c(seq(0, 200000, by = 50000), 10000000)
df$Mileage = cut(df$Mileage, breaks = breaks)
summary(df$Mileage)

# 6. Variable Model.Year --> convertie en ancienneté
summary(df$Model.Year)
ggp = ggplot(df, aes(Model.Year))+
  geom_histogram(bins = 100, fill = "blue", color = "black", alpha = 0.7)+
  theme_light()
ggp

# On transforme l'année en ancienneté
df$Model.Year = 2019 - df$Model.Year
cor(df$Model.Year, log(df$Price))

ggp = ggplot(df, aes(x = Model.Year, y = log(Price))) +
  geom_bin2d(bins = 30)+
  geom_smooth(method = "lm", color = "blue")
ggp

df$Model.Year = as.factor(df$Model.Year)

# 7. Variable Engine.Capacity (discrétisation)
summary(df$Engine.Capacity)
ggp = ggplot(df, aes(Engine.Capacity))+
  geom_histogram(bins = 100, fill = "blue", color = "black", alpha = 0.7)+
  theme_light()
ggp

cor(df$Engine.Capacity, log(df$Price))

breaks = c(0,701,951,1105,1350,1550,1650,1850,10000)
df$Engine.Capacity = cut(df$Engine.Capacity, breaks)
df$Engine.Capacity = as.factor(df$Engine.Capacity)
summary(df$Engine.Capacity)

# 8. Séparation en jeu d’entraînement et de test
set.seed(1)
train = sample(nrow(df), nrow(df)*0.8)
df_train = df[train, ]
df_test  = df[-train, ]

# 9. Modélisation

###############################################################################
# 9.1. Modèle de Régression Linéaire
###############################################################################
model_lm = lm(log(Price) ~ ., data = df_train)
summary(model_lm)

# Analyse des coefficients
cat("Importance des variables - Régression linéaire :\n")
print(summary(model_lm)$coefficients)

# Prédictions (log) sur le jeu de test
pred_lm_log = predict(model_lm, newdata = df_test)

#----------------------
# METRIQUES EN ÉCHELLE LOG
#----------------------
# RMSE (log)
rmse_lm_log = sqrt(mean((log(df_test$Price) - pred_lm_log)^2))
# MAE (log)
mae_lm_log  = mean(abs(log(df_test$Price) - pred_lm_log))
# R^2 (log)
r2_lm_log   = 1 - sum((log(df_test$Price) - pred_lm_log)^2) / 
  sum((log(df_test$Price) - mean(log(df_test$Price)))^2)

#----------------------
# METRIQUES EN ÉCHELLE RÉELLE
#----------------------
# On exponentie les prédictions pour revenir à l’échelle du prix
pred_lm = exp(pred_lm_log)

# MSE
mse_lm = mean((df_test$Price - pred_lm)^2)
# RMSE
rmse_lm = sqrt(mse_lm)
# MAE
mae_lm = mean(abs(df_test$Price - pred_lm))
# MAPE (en %)
mape_lm = mean(abs(df_test$Price - pred_lm) / df_test$Price) * 100
# R^2
sst_lm  = sum((df_test$Price - mean(df_test$Price))^2)
sse_lm  = sum((df_test$Price - pred_lm)^2)
r2_lm   = 1 - sse_lm/sst_lm

# R^2 ajusté (calcul sur le jeu de test, attention à l’interprétation)
n_test_lm = nrow(df_test)
p_lm      = length(coef(model_lm)) - 1  # Nb de variables explicatives
adj_r2_lm = 1 - (1 - r2_lm) * ((n_test_lm - 1)/(n_test_lm - p_lm - 1))

cat("\nPerformance - Régression linéaire (test set) :\n")
cat(" -- Échelle log --\n")
cat("    RMSE(log) :", rmse_lm_log, "\n")
cat("    MAE(log)  :", mae_lm_log,  "\n")
cat("    R^2(log)  :", r2_lm_log,   "\n")
cat(" -- Échelle réelle --\n")
cat("    MSE       :", mse_lm,      "\n")
cat("    RMSE      :", rmse_lm,     "\n")
cat("    MAE       :", mae_lm,      "\n")
cat("    MAPE(%)   :", mape_lm,     "\n")
cat("    R^2       :", r2_lm,       "\n")
cat("    R^2 adj   :", adj_r2_lm,   "\n")

# Visualisation
ggplot(data = data.frame(Observed = log(df_test$Price), Predicted = pred_lm_log), 
       aes(x = Observed, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Régression Linéaire: Observé vs Prédit (log-scale)", 
       x = "Valeurs Observées (log)", 
       y = "Valeurs Prédites (log)") +
  theme_minimal()


###############################################################################
# 9.2. Modèle Random Forest
###############################################################################
set.seed(1)
model_rf = randomForest(log(Price) ~ ., data = df_train, 
                        ntree = 10, 
                        mtry = 2, 
                        importance = TRUE)
print(model_rf)

# Importance des variables
cat("\nImportance des variables - Random Forest :\n")
print(importance(model_rf))

# Visualisation de la profondeur moyenne des arbres
plot_min_depth_distribution(model_rf)

# Prédictions (log) sur les données de test
pred_rf_log = predict(model_rf, newdata = df_test)

#----------------------
# METRIQUES EN ÉCHELLE LOG
#----------------------
rmse_rf_log = sqrt(mean((log(df_test$Price) - pred_rf_log)^2))
mae_rf_log  = mean(abs(log(df_test$Price) - pred_rf_log))
r2_rf_log   = 1 - sum((log(df_test$Price) - pred_rf_log)^2) /
  sum((log(df_test$Price) - mean(log(df_test$Price)))^2)

#----------------------
# METRIQUES EN ÉCHELLE RÉELLE
#----------------------
pred_rf = exp(pred_rf_log)

mse_rf  = mean((df_test$Price - pred_rf)^2)
rmse_rf = sqrt(mse_rf)
mae_rf  = mean(abs(df_test$Price - pred_rf))
mape_rf = mean(abs(df_test$Price - pred_rf) / df_test$Price) * 100

sst_rf  = sum((df_test$Price - mean(df_test$Price))^2)
sse_rf  = sum((df_test$Price - pred_rf)^2)
r2_rf   = 1 - sse_rf/sst_rf

# On peut calculer un pseudo R^2 ajusté de la même façon, si besoin
n_test_rf  = nrow(df_test)
p_rf       = length(df_train) - 1  # (approximation)
adj_r2_rf  = 1 - (1 - r2_rf) * ((n_test_rf - 1)/(n_test_rf - p_rf - 1))

cat("\nPerformance - Random Forest (test set) :\n")
cat(" -- Échelle log --\n")
cat("    RMSE(log) :", rmse_rf_log, "\n")
cat("    MAE(log)  :", mae_rf_log,  "\n")
cat("    R^2(log)  :", r2_rf_log,   "\n")
cat(" -- Échelle réelle --\n")
cat("    MSE       :", mse_rf,      "\n")
cat("    RMSE      :", rmse_rf,     "\n")
cat("    MAE       :", mae_rf,      "\n")
cat("    MAPE(%)   :", mape_rf,     "\n")
cat("    R^2       :", r2_rf,       "\n")
cat("    R^2 adj   :", adj_r2_rf,   "\n")

# Visualisation
ggplot(data = data.frame(Observed = log(df_test$Price), Predicted = pred_rf_log), 
       aes(x = Observed, y = Predicted)) +
  geom_point(color = "green", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Random Forest: Observé vs Prédit (log-scale)", 
       x = "Valeurs Observées (log)", 
       y = "Valeurs Prédites (log)") +
  theme_minimal()


###############################################################################
# 9.3. Modèle SVR (Support Vector Regression)
###############################################################################
set.seed(42)
model_svr = svm(log(Price) ~ ., data = df_train, kernel = "radial")
summary(model_svr)

pred_svr_log = predict(model_svr, newdata = df_test)

#----------------------
# METRIQUES EN ÉCHELLE LOG
#----------------------
rmse_svr_log = sqrt(mean((log(df_test$Price) - pred_svr_log)^2))
mae_svr_log  = mean(abs(log(df_test$Price) - pred_svr_log))
r2_svr_log   = 1 - sum((log(df_test$Price) - pred_svr_log)^2) /
  sum((log(df_test$Price) - mean(log(df_test$Price)))^2)

#----------------------
# METRIQUES EN ÉCHELLE RÉELLE
#----------------------
pred_svr = exp(pred_svr_log)

mse_svr  = mean((df_test$Price - pred_svr)^2)
rmse_svr = sqrt(mse_svr)
mae_svr  = mean(abs(df_test$Price - pred_svr))
mape_svr = mean(abs(df_test$Price - pred_svr) / df_test$Price) * 100

sst_svr  = sum((df_test$Price - mean(df_test$Price))^2)
sse_svr  = sum((df_test$Price - pred_svr)^2)
r2_svr   = 1 - sse_svr/sst_svr

n_test_svr  = nrow(df_test)
p_svr       = length(df_train) - 1  # approximation
adj_r2_svr  = 1 - (1 - r2_svr) * ((n_test_svr - 1)/(n_test_svr - p_svr - 1))

cat("\nPerformance - SVR (test set) :\n")
cat(" -- Échelle log --\n")
cat("    RMSE(log) :", rmse_svr_log, "\n")
cat("    MAE(log)  :", mae_svr_log,  "\n")
cat("    R^2(log)  :", r2_svr_log,   "\n")
cat(" -- Échelle réelle --\n")
cat("    MSE       :", mse_svr,      "\n")
cat("    RMSE      :", rmse_svr,     "\n")
cat("    MAE       :", mae_svr,      "\n")
cat("    MAPE(%)   :", mape_svr,     "\n")
cat("    R^2       :", r2_svr,       "\n")
cat("    R^2 adj   :", adj_r2_svr,   "\n")

ggplot(data = data.frame(Observed = log(df_test$Price), Predicted = pred_svr_log), 
       aes(x = Observed, y = Predicted)) +
  geom_point(color = "purple", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "SVR: Observé vs Prédit (log-scale)", 
       x = "Valeurs Observées (log)", 
       y = "Valeurs Prédites (log)") +
  theme_minimal()


###############################################################################
# 9.4. Modèle de Gradient Boosting (xgboost)
###############################################################################
# Préparation des matrices
train_matrix = model.matrix(log(Price) ~ . - 1, data = df_train)
test_matrix  = model.matrix(log(Price) ~ . - 1, data = df_test)
train_labels = log(df_train$Price)
test_labels  = log(df_test$Price)

dtrain = xgb.DMatrix(data = train_matrix, label = train_labels)
dtest  = xgb.DMatrix(data = test_matrix,  label = test_labels)

# Entraînement
set.seed(42)
model_xgb = xgboost(data = dtrain, 
                    max_depth = 5, 
                    eta = 0.1, 
                    nrounds = 500, 
                    objective = "reg:squarederror", 
                    verbose = 0)

# Prédictions (log)
pred_xgb_log = predict(model_xgb, newdata = dtest)

#----------------------
# METRIQUES EN ÉCHELLE LOG
#----------------------
rmse_xgb_log = sqrt(mean((test_labels - pred_xgb_log)^2))
mae_xgb_log  = mean(abs(test_labels - pred_xgb_log))
r2_xgb_log   = 1 - sum((test_labels - pred_xgb_log)^2) /
  sum((test_labels - mean(test_labels))^2)

#----------------------
# METRIQUES EN ÉCHELLE RÉELLE
#----------------------
pred_xgb     = exp(pred_xgb_log)
mse_xgb      = mean((df_test$Price - pred_xgb)^2)
rmse_xgb     = sqrt(mse_xgb)
mae_xgb      = mean(abs(df_test$Price - pred_xgb))
mape_xgb     = mean(abs(df_test$Price - pred_xgb) / df_test$Price) * 100

sst_xgb      = sum((df_test$Price - mean(df_test$Price))^2)
sse_xgb      = sum((df_test$Price - pred_xgb)^2)
r2_xgb       = 1 - sse_xgb/sst_xgb

n_test_xgb   = nrow(df_test)
p_xgb        = length(df_train) - 1  # approximation
adj_r2_xgb   = 1 - (1 - r2_xgb) * ((n_test_xgb - 1)/(n_test_xgb - p_xgb - 1))

cat("\nPerformance - xgboost (test set) :\n")
cat(" -- Échelle log --\n")
cat("    RMSE(log) :", rmse_xgb_log, "\n")
cat("    MAE(log)  :", mae_xgb_log,  "\n")
cat("    R^2(log)  :", r2_xgb_log,   "\n")
cat(" -- Échelle réelle --\n")
cat("    MSE       :", mse_xgb,      "\n")
cat("    RMSE      :", rmse_xgb,     "\n")
cat("    MAE       :", mae_xgb,      "\n")
cat("    MAPE(%)   :", mape_xgb,     "\n")
cat("    R^2       :", r2_xgb,       "\n")
cat("    R^2 adj   :", adj_r2_xgb,   "\n")

ggplot(data = data.frame(Observed = test_labels, Predicted = pred_xgb_log), 
       aes(x = Observed, y = Predicted)) +
  geom_point(color = "orange", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "xgboost: Observé vs Prédit (log-scale)", 
       x = "Valeurs Observées (log)", 
       y = "Valeurs Prédites (log)") +
  theme_minimal()

###############################################################################
# 10. Comparaison des performances (focus sur RMSE et R^2 en log + réel)
###############################################################################
cat("\n\n=== COMPARAISON DES MODÈLES ===\n")

cat("\n--- [Échelle LOG] ---\n")
cat("Régression linéaire - RMSE:", rmse_lm_log, " | R^2:", r2_lm_log, "\n")
cat("Random Forest         - RMSE:", rmse_rf_log, " | R^2:", r2_rf_log, "\n")
cat("SVR                  - RMSE:", rmse_svr_log, " | R^2:", r2_svr_log, "\n")
cat("xgboost              - RMSE:", rmse_xgb_log, " | R^2:", r2_xgb_log, "\n")

cat("\n--- [Échelle RÉELLE] ---\n")
cat("Régression linéaire - RMSE:", rmse_lm,    " | R^2:", r2_lm,    "\n")
cat("Random Forest       - RMSE:", rmse_rf,    " | R^2:", r2_rf,    "\n")
cat("SVR                - RMSE:", rmse_svr,    " | R^2:", r2_svr,   "\n")
cat("xgboost            - RMSE:", rmse_xgb,    " | R^2:", r2_xgb,   "\n")



###############################################################################
# 1) Créer le data.frame des métriques
###############################################################################
df_metrics = data.frame(
  Model = c("Régression Linéaire", "Random Forest", "SVR", "xgboost"),
  RMSE  = c(rmse_lm, rmse_rf, rmse_svr, rmse_xgb),
  MAPE  = c(mape_lm, mape_rf, mape_svr, mape_xgb),
  R2    = c(r2_lm, r2_rf, r2_svr, r2_xgb)
)

###############################################################################
# 2) Afficher le tableau dans un R Markdown 
###############################################################################
#install.packages("pander") # si besoin
library(pander)

# Optionnel : vous pouvez régler le nombre de décimales
panderOptions('digits', 3)  # pour n'afficher que 3 décimales

# Affichage du tableau
pander(
  df_metrics,
  style = 'rmarkdown',
  caption = 'Comparaison des performances des modèles (échelle réelle)'
)


#!!
###############################################################################
# 11. Validation Croisée Manuelle - Modèles                                  #
###############################################################################

# Fonction pour calculer les métriques de performance
calculate_metrics <- function(actual, predicted) {
  mse  <- mean((actual - predicted)^2)
  rmse <- sqrt(mse)
  mae  <- mean(abs(actual - predicted))
  r2   <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  return(list(MSE = mse, RMSE = rmse, MAE = mae, R2 = r2))
}

# Fonction de validation croisée manuelle
cross_validation <- function(model_func, data, k = 5) {
  set.seed(42)
  folds <- cut(seq(1, nrow(data)), breaks = k, labels = FALSE)
  metrics_list <- list()
  
  for (i in 1:k) {
    test_indices  <- which(folds == i, arr.ind = TRUE)
    test_data     <- data[test_indices, ]
    train_data    <- data[-test_indices, ]
    
    model <- model_func(train_data)
    predictions_log <- predict(model, newdata = test_data)
    predictions <- exp(predictions_log)
    
    metrics <- calculate_metrics(test_data$Price, predictions)
    metrics_list[[i]] <- metrics
  }
  
  avg_metrics <- sapply(metrics_list, function(x) unlist(x))
  avg_metrics <- rowMeans(avg_metrics)
  return(avg_metrics)
}

###############################################################################
# 11.1. Régression Linéaire - Validation croisée
###############################################################################
model_lm_func <- function(train_data) {
  lm(log(Price) ~ ., data = train_data)
}

lm_cv_metrics <- cross_validation(model_lm_func, df_train)
cat("\nValidation croisée - Régression Linéaire :\n")
print(lm_cv_metrics)

###############################################################################
# 11.2. Random Forest - Validation croisée
###############################################################################
model_rf_func <- function(train_data) {
  randomForest(log(Price) ~ ., data = train_data, ntree = 10, mtry = 2)
}

rf_cv_metrics <- cross_validation(model_rf_func, df_train)
cat("\nValidation croisée - Random Forest :\n")
print(rf_cv_metrics)

###############################################################################
# 11.3. SVR - Validation croisée
###############################################################################
model_svr_func <- function(train_data) {
  svm(log(Price) ~ ., data = train_data, kernel = "radial")
}

svr_cv_metrics <- cross_validation(model_svr_func, df_train)
cat("\nValidation croisée - SVR :\n")
print(svr_cv_metrics)

###############################################################################
# 11.4. XGBoost - Validation croisée
###############################################################################
model_xgb_func <- function(train_data) {
  train_matrix <- model.matrix(log(Price) ~ . - 1, data = train_data)
  dtrain <- xgb.DMatrix(data = train_matrix, label = log(train_data$Price))
  xgboost(data = dtrain, max_depth = 5, eta = 0.1, nrounds = 100, objective = "reg:squarederror", verbose = 0)
}

cross_validation_xgb <- function(model_func, data, k = 5) {
  set.seed(42)
  folds <- cut(seq(1, nrow(data)), breaks = k, labels = FALSE)
  metrics_list <- list()
  
  for (i in 1:k) {
    test_indices  <- which(folds == i, arr.ind = TRUE)
    test_data     <- data[test_indices, ]
    train_data    <- data[-test_indices, ]
    
    model <- model_func(train_data)
    test_matrix <- model.matrix(log(Price) ~ . - 1, data = test_data)
    predictions_log <- predict(model, newdata = xgb.DMatrix(test_matrix))
    predictions <- exp(predictions_log)
    
    metrics <- calculate_metrics(test_data$Price, predictions)
    metrics_list[[i]] <- metrics
  }
  
  avg_metrics <- sapply(metrics_list, function(x) unlist(x))
  avg_metrics <- rowMeans(avg_metrics)
  return(avg_metrics)
}

xgb_cv_metrics <- cross_validation_xgb(model_xgb_func, df_train)
cat("\nValidation croisée - XGBoost :\n")
print(xgb_cv_metrics)

###############################################################################
# 11.5. Comparaison des modèles avec Validation croisée
###############################################################################
df_cv_results <- data.frame(
  Model = c("Régression Linéaire", "Random Forest", "SVR", "XGBoost"),
  MSE   = c(lm_cv_metrics["MSE"], rf_cv_metrics["MSE"], svr_cv_metrics["MSE"], xgb_cv_metrics["MSE"]),
  RMSE  = c(lm_cv_metrics["RMSE"], rf_cv_metrics["RMSE"], svr_cv_metrics["RMSE"], xgb_cv_metrics["RMSE"]),
  MAE   = c(lm_cv_metrics["MAE"], rf_cv_metrics["MAE"], svr_cv_metrics["MAE"], xgb_cv_metrics["MAE"]),
  R2    = c(lm_cv_metrics["R2"], rf_cv_metrics["R2"], svr_cv_metrics["R2"], xgb_cv_metrics["R2"])
)

print(df_cv_results)

###############################################################################
# 12. Optimisation des Hyperparamètres - XGBoost avec Grid Search             #
###############################################################################

# Définir la grille de recherche des hyperparamètres
xgb_grid <- expand.grid(
  max_depth = c(3, 5),
  eta = c(0.01, 0.1),
  nrounds = c(100, 500),
  colsample_bytree = c(0.7, 1),
  min_child_weight = c(1, 3),
  subsample = c(0.7, 1)
)

# Fonction pour entraîner XGBoost avec des hyperparamètres donnés
train_xgb_model <- function(params, train_data) {
  train_matrix <- model.matrix(log(Price) ~ . - 1, data = train_data)
  dtrain <- xgb.DMatrix(data = train_matrix, label = log(train_data$Price))
  
  model <- xgboost(
    data = dtrain,
    max_depth = params$max_depth,
    eta = params$eta,
    nrounds = params$nrounds,
    colsample_bytree = params$colsample_bytree,
    min_child_weight = params$min_child_weight,
    subsample = params$subsample,
    objective = "reg:squarederror",
    verbose = 0
  )
  return(model)
}

# Grid Search avec validation croisée manuelle
best_rmse <- Inf
best_params <- NULL

for (i in 1:nrow(xgb_grid)) {
  params <- xgb_grid[i, ]
  model_func <- function(train_data) train_xgb_model(params, train_data)
  metrics <- cross_validation_xgb(model_func, df_train)
  
  if (metrics["RMSE"] < best_rmse) {
    best_rmse <- metrics["RMSE"]
    best_params <- params
  }
  cat("Essai ", i, "/", nrow(xgb_grid), " - RMSE :", metrics["RMSE"], "\n")
}

cat("\nMeilleurs hyperparamètres XGBoost :\n")
print(best_params)
cat("RMSE obtenu :", best_rmse, "\n")

# Entraînement du modèle XGBoost optimisé
model_xgb_optimized <- train_xgb_model(best_params, df_train)

# Évaluation sur le jeu de test
test_matrix <- model.matrix(log(Price) ~ . - 1, data = df_test)
dtest <- xgb.DMatrix(data = test_matrix)
pred_xgb_opt_log <- predict(model_xgb_optimized, newdata = dtest)
pred_xgb_opt <- exp(pred_xgb_opt_log)

# Calcul des métriques (RMSE, MAPE, R²)
calculate_metrics <- function(actual, predicted) {
  mse <- mean((actual - predicted)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(actual - predicted))
  mape <- mean(abs((actual - predicted) / actual)) * 100
  r2 <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  return(list(RMSE = rmse, MAPE = mape, R2 = r2))
}

optimized_metrics <- calculate_metrics(df_test$Price, pred_xgb_opt)
cat("\nPerformances du XGBoost optimisé :\n")
cat("RMSE :", optimized_metrics$RMSE, "\n")
cat("MAPE :", optimized_metrics$MAPE, "%\n")
cat("R² :", optimized_metrics$R2, "\n")


###############################################################################
# 9.4.bis Modèle de Gradient Boosting (xgboost)
###############################################################################
# Préparation des matrices
train_matrix = model.matrix(log(Price) ~ . - 1, data = df_train)
test_matrix  = model.matrix(log(Price) ~ . - 1, data = df_test)
train_labels = log(df_train$Price)
test_labels  = log(df_test$Price)

dtrain = xgb.DMatrix(data = train_matrix, label = train_labels)
dtest  = xgb.DMatrix(data = test_matrix,  label = test_labels)

# Entraînement
set.seed(42)
model_xgb = xgboost(data = dtrain, 
                    max_depth = 15, 
                    eta = 0.1, 
                    nrounds = 1000, 
                    colsample_bytree = 0.7,
                    subsample = 0.7,
                    objective = "reg:squarederror", 
                    verbose = 0)

# Prédictions (log)
pred_xgb_log = predict(model_xgb, newdata = dtest)

#----------------------
# METRIQUES EN ÉCHELLE LOG
#----------------------
rmse_xgb_log = sqrt(mean((test_labels - pred_xgb_log)^2))
mae_xgb_log  = mean(abs(test_labels - pred_xgb_log))
r2_xgb_log   = 1 - sum((test_labels - pred_xgb_log)^2) /
  sum((test_labels - mean(test_labels))^2)

#----------------------
# METRIQUES EN ÉCHELLE RÉELLE
#----------------------
pred_xgb     = exp(pred_xgb_log)
mse_xgb      = mean((df_test$Price - pred_xgb)^2)
rmse_xgb     = sqrt(mse_xgb)
mae_xgb      = mean(abs(df_test$Price - pred_xgb))
mape_xgb     = mean(abs(df_test$Price - pred_xgb) / df_test$Price) * 100

sst_xgb      = sum((df_test$Price - mean(df_test$Price))^2)
sse_xgb      = sum((df_test$Price - pred_xgb)^2)
r2_xgb       = 1 - sse_xgb/sst_xgb

n_test_xgb   = nrow(df_test)
p_xgb        = length(df_train) - 1  # approximation
adj_r2_xgb   = 1 - (1 - r2_xgb) * ((n_test_xgb - 1)/(n_test_xgb - p_xgb - 1))

cat("\nPerformance - xgboost (test set) :\n")
cat(" -- Échelle log --\n")
cat("    RMSE(log) :", rmse_xgb_log, "\n")
cat("    MAE(log)  :", mae_xgb_log,  "\n")
cat("    R^2(log)  :", r2_xgb_log,   "\n")
cat(" -- Échelle réelle --\n")
cat("    MSE       :", mse_xgb,      "\n")
cat("    RMSE      :", rmse_xgb,     "\n")
cat("    MAE       :", mae_xgb,      "\n")
cat("    MAPE(%)   :", mape_xgb,     "\n")
cat("    R^2       :", r2_xgb,       "\n")
cat("    R^2 adj   :", adj_r2_xgb,   "\n")

ggplot(data = data.frame(Observed = test_labels, Predicted = pred_xgb_log), 
       aes(x = Observed, y = Predicted)) +
  geom_point(color = "pink", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "xgboost: Observé vs Prédit (log-scale)", 
       x = "Valeurs Observées (log)", 
       y = "Valeurs Prédites (log)") +
  theme_minimal()

###############################################################################
# 13. Interprétation du modèle XGBoost optimisé (Alternative sans SHAP)      #
###############################################################################

# Importance des variables
importance_matrix <- xgb.importance(model = model_xgb)
print(importance_matrix)

# Visualisation de l'importance des variables
xgb.plot.importance(importance_matrix, top_n = 10, 
                    main = "Importance des Variables - XGBoost")

# Affichage des arbres de décision (premier arbre)
xgb.plot.tree(model = model_xgb_optimized, trees = 0, 
              main = "Premier Arbre de Décision - XGBoost")

###############################################################################
# 14. Analyse Partielle des Dépendances (PDP)                                #
###############################################################################

# Chargement de la librairie pour PDP
library(pdp)

# Analyse de la variable la plus importante
top_feature <- importance_matrix$Feature[1]

# Création de la PDP pour la variable la plus influente
pdp_plot <- partial(model_xgb_optimized, pred.var = top_feature, 
                    train = df_train, grid.resolution = 20, prob = TRUE)

# Visualisation de la PDP
plotPartial(pdp_plot, main = paste("PDP pour", top_feature),
            xlab = top_feature, ylab = "Effet sur le log(Prix)")

###############################################################################
# 15. Interaction entre variables (ICE Plot)                                 #
###############################################################################

# Plot des effets individuels conditionnels
ice_plot <- partial(model_xgb_optimized, pred.var = top_feature, 
                    train = df_train, grid.resolution = 20, ice = TRUE)

# Visualisation de l'ICE Plot
plotPartial(ice_plot, main = paste("ICE Plot pour", top_feature),
            xlab = top_feature, ylab = "Effet individuel sur le log(Prix)")

cat("\nInterprétation terminée : Importance des variables et PDP/ICE")
