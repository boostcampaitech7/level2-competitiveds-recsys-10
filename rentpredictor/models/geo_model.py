import pandas as pd
import lightgbm as lgb
from .model import Model
from ..preprocessor import Preprocessor

class GeoModel(Model):
    """각종 지리정보가 feature로 추가된 모델입니다.
    """

    def set_data(self, dataframes: dict[str, pd.DataFrame]) -> None:
        """각종 데이터를 설정합니다.

        Parameters
        ----------
        dataframes : dict[str, pd.DataFrame]
            학습 및 테스트 데이터, 그리고 그 외 정보를 포함한 딕셔너리입니다.
        """
        self.dataframes = dataframes

    def preprocess(self) -> None:
        """데이터를 전처리합니다.
        """
        preprocessor = Preprocessor(self.dataframes)
        preprocessor.remove_unnecessary_locations(0.1)
        preprocessor.add_location_id()

        # 금리를 추가합니다.
        # preprocessor.add_interest_rate()

        # 공원 종류별로 최소 거리를 추가합니다.
        # preprocessor.add_min_distance_to_park()
        # preprocessor.add_min_distance_to_park(min_area=10000)
        # preprocessor.add_min_distance_to_park(min_area=100000)
        # preprocessor.add_min_distance_to_park(min_area=1000000)

        # 학교 종류별로 최소 거리를 추가합니다.
        # preprocessor.add_min_distance_to_school('elementary')
        # preprocessor.add_min_distance_to_school('middle')
        # preprocessor.add_min_distance_to_school('high')
        # preprocessor.add_min_distance_to_school('all')

        # 지하철역까지의 최소 거리를 추가합니다.
        # preprocessor.add_min_distance_to_subway()
        
        # 반경 0.1 내에 있는 지하철역, 공원, 학교의 개수를 추가합니다.
        preprocessor.add_locations_within_radius('subway', 0.1)
        preprocessor.add_locations_within_radius('park', 0.1)
        preprocessor.add_locations_within_radius('school', 0.1)

        # 반경 0.01 내에 있는 지하철역, 공원, 학교의 개수를 추가합니다.
        preprocessor.add_locations_within_radius('subway', 0.01)
        preprocessor.add_locations_within_radius('park', 0.01)
        preprocessor.add_locations_within_radius('school', 0.01)

        preprocessor.drop_columns(['location_id', 'contract_day', 'contract_type', 'age'])

        self.train_df = preprocessor.get_train_df()
        self.test_df = preprocessor.get_test_df()

    def fit(self) -> None:
        """학습 데이터를 통해 모델을 학습합니다.
        """
        X_train, y_train = self.train_df.drop(columns=['deposit']), self.train_df['deposit']
        X_val, y_val = self.test_df.drop(columns=['deposit']), self.test_df['deposit']
        
        train_dataset = lgb.Dataset(X_train, y_train)
        val_dataset = lgb.Dataset(X_val, y_val)
        self.lgb_model = lgb.train(
            params={
                'objective': 'regression',
                'metric': 'mae',
                'num_leaves': 63,
            },
            train_set=train_dataset,
            valid_sets=[
                train_dataset,
                val_dataset,
            ],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.log_evaluation(period=100),
            ],
            num_boost_round=10000,
        )
        lgb.plot_importance(self.lgb_model)
        import matplotlib.pyplot as plt
        plt.savefig('feature_importance.png', bbox_inches='tight')


    def predict(self) -> pd.Series:
        """테스트 데이터에 대해 예측을 수행합니다.

        Returns
        -------
        pd.Series
            테스트 데이터에 대한 예측한 결과입니다.
        """
        X_test = self.test_df.drop(columns=['deposit'])
        y_test_pred = self.lgb_model.predict(X_test)
        y_test_pred = pd.Series(y_test_pred)
        return y_test_pred