import optuna
import sklearn.datasets
import sklearn.ensemble
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import sklearn.model_selection
import xgboost as xgb
from catboost import CatBoostRegressor

class ParameterTuner:
    """하이퍼 파라미터를 최적화 하는 class입니다. 
    """
    def __init__(self,model : str,X_train : pd.DataFrame,y_train : pd.Series,X_val : pd.DataFrame,y_val : pd.Series) -> None:
        """class를 초기화 하는 함수입니다.

        Parameters
        ----------
        model : str
            입력받는 모델의 이름입니다. 현재 lgb,XGBoost,CatBoost가 존재합니다.
        X : pd.DataFrame
            학습을 진행하는 데이터입니다.
        y : pd.Serise
            학습 데이터에 관한 target 값입니다.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = model

    def objective(self,trial : optuna.Trial) -> float:
        """Optuna를 사용하여 하이퍼 파라미터 최적화를 수행하는 함수입니다.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna가 각 반복에서 전달하는 trial 객체로, 하이퍼 파라미터를 제안하는 역할을 합니다.

        Returns
        -------
        float
            검증 데이터에 대한 Mean Absolute Error (MAE)를 반환합니다
        """
        if self.model == "lgb":
            param = {
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.1),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
                "num_leaves": trial.suggest_int("num_leaves", 20, 3000),
                "min_child_samples" : trial.suggest_int('min_child_samples',1,300),
                "subsample" : trial.suggest_uniform('subsample',0.6,1.0),
                "colsample_bytree" : trial.suggest_uniform('colsample_bytree',0.6,1.0),
                "random_state" : 42
            }
            clf = lgb.LGBMRegressor(**param,force_col_wise=True)
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_val)

        elif self.model == "XGBoost":
            param = {
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3,log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            }
            # XGBoost 모델 학습
            clf = xgb.XGBRegressor(**param)
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_val)


        elif self.model == "CatBoost":
            param = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "iterations": int(trial.suggest_int("iterations", 100, 1000, log=True)),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "border_count": trial.suggest_int("border_count", 32, 255),
            }

            # CatBoost 모델 학습
            clf = CatBoostRegressor(**param, verbose=0)  # verbose=0으로 출력을 억제
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_val)

        return mean_absolute_error(self.y_val,y_pred)
        

    def tune(self) -> optuna.Trial:
        """Optuna를 사용하여 하이퍼 파라미터 튜닝을 수행하는 함수입니다.

        주어진 모델에 대해 하이퍼 파라미터를 자동으로 탐색하며, 최적의 파라미터를 반환합니다.
        Optuna의 `study.optimize` 메서드를 사용하여 지정된 횟수만큼 실험을 진행하고,
        가장 성능이 좋은 하이퍼 파라미터를 찾습니다.

        Returns
        -------
        dict
            최적의 하이퍼 파라미터를 key-value 쌍으로 반환합니다.

        Example
        -------
        >>> parameter_tuner = ParameterTune('lgb', X, y)
        >>> best_params = parameter_tuner.tune()
        >>> print(best_params)
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective,n_trials=10)

        trial = study.best_trial
        print(f"MAE : {trial.value}")

        return trial.params