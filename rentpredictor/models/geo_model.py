import os
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
from .model import Model
from ..preprocessor import Preprocessor

class GeoModel(Model):
    """각종 지리정보가 feature로 추가된 모델입니다.
    """

    def preprocess(self, preprocessor: Preprocessor) -> None:
        """지리 관련 feature를 추가합니다.

        Parameters
        ----------
        preprocessor : Preprocessor
            전처리될 데이터를 담고 있는 Preprocessor 객체입니다.
        """
        preprocessor.remove_unnecessary_locations(0.1)
        preprocessor.add_location_id()

        # 금리를 추가합니다.
        preprocessor.add_interest_rate()

        # 공원 종류별로 최소 거리를 추가합니다.
        preprocessor.add_min_distance_to_park()
        preprocessor.add_min_distance_to_park(min_area=10000)
        preprocessor.add_min_distance_to_park(min_area=100000)
        preprocessor.add_min_distance_to_park(min_area=1000000)

        # 학교 종류별로 최소 거리를 추가합니다.
        preprocessor.add_min_distance_to_school('elementary')
        preprocessor.add_min_distance_to_school('middle')
        preprocessor.add_min_distance_to_school('high')
        preprocessor.add_min_distance_to_school('all')

        # 지하철역까지의 최소 거리를 추가합니다.
        preprocessor.add_min_distance_to_subway()

        # 반경 0.1 내에 있는 지하철역, 공원, 학교의 개수를 추가합니다.
        preprocessor.add_locations_within_radius('subway', 0.1)
        preprocessor.add_locations_within_radius('park', 0.1)
        preprocessor.add_locations_within_radius('school', 0.01)
        preprocessor.add_locations_within_radius('subway', 0.01)
        preprocessor.add_locations_within_radius('park', 0.01)
        preprocessor.add_locations_within_radius('school', 0.01)

        # 위도, 경도를 binning하여 추가합니다.
        preprocessor.add_latitude_bin(0.1)
        preprocessor.add_longitude_bin(0.1)
        preprocessor.add_latitude_bin(0.01)
        preprocessor.add_longitude_bin(0.01)

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """모델을 학습합니다.

        Parameters
        ----------
        X : pd.DataFrame
            학습 데이터입니다.
        y : pd.Series
            학습 데이터의 target 값입니다.
        X_val : pd.DataFrame, optional
            검증 데이터입니다.
            early stopping을 위한 것으로, 선택적으로 제공합니다.
        y_val : pd.Series, optional
            검증 데이터의 target 값입니다.
            early stopping을 위한 것으로, 선택적으로 제공합니다.
        """
        if X_val is None and y_val is None:
            X_val, y_val = X, y

        X_all = pd.concat([X, X_val], axis=0)
        X_all.drop(columns=['contract_datetime'], inplace=True, errors='ignore')
        X, X_val = X_all.iloc[:len(X)], X_all.iloc[len(X):]
        
        if not os.path.exists('model.txt'):
            self.lgb_model = lgb.train(
                params={
                    'objective': 'regression',
                    'metric': 'mae',
                    'num_leaves': 63,
                    'seed': 42,
                    'verbose': -1,
                },
                train_set=lgb.Dataset(X, y),
                valid_sets=[
                    lgb.Dataset(X, y),
                    lgb.Dataset(X_val, y_val),
                ],
                valid_names=['train', 'val'],
                callbacks=[
                    lgb.log_evaluation(period=100),
                    lgb.early_stopping(stopping_rounds=100),
                ],
                num_boost_round=1500,
            )
            self.lgb_model.save_model('model.txt')
        else:
            self.lgb_model = lgb.Booster(model_file='model.txt')

        lgb.plot_importance(self.lgb_model, importance_type='gain')
        plt.savefig('feature_importance.png', bbox_inches='tight')

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """테스트 데이터에 대해 예측을 수행합니다.

        메서드를 호출하기 전, fit 메서드를 먼저 호출해야합니다.

        Parameters
        ----------
        X : pd.DataFrame
            예측을 수행할 테스트데이터입니다.

        Returns
        -------
        pd.Series
            테스트데이터에 대한 예측한 결과입니다.
        """
        X.drop(columns=['contract_datetime'], inplace=True, errors='ignore')
        y_pred = self.lgb_model.predict(X)
        return pd.Series(y_pred)