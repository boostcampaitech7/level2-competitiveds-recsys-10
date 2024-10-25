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
        preprocessor.add_locations_within_radius('school', 0.1)

        preprocessor.add_transfer_stations_within_radius(0.1)

        preprocessor.add_latitude_bin(0.1)
        preprocessor.add_longitude_bin(0.1)

        self.train_df = preprocessor.get_train_df()
        self.test_df = preprocessor.get_test_df()

    def fit(self) -> None:
        """학습 데이터를 통해 모델을 학습합니다.
        """
        X_train, y_train = self.train_df.drop(columns=['deposit']), self.train_df['deposit']
        X_val, y_val = self.test_df.drop(columns=['deposit']), self.test_df['deposit']
        
        train_dataset = lgb.Dataset(X_train, y_train)
        val_dataset = lgb.Dataset(X_val, y_val)
        params = {
            'learning_rate': 0.07143490991704106, 'n_estimators': 1418, 'num_leaves': 116, 
            'max_depth': 14, 'min_child_weight': 18, 'subsample': 0.7903627891322709, 
            'colsample_bytree': 0.892700154945412, 'lambda_l1': 5.5916562635855494e-08, 
            'lambda_l2': 0.00014233586207456367, 'min_split_gain': 0.7602166984240577
        }
        self.lgb_model = lgb.train(
            params=params,
            train_set=train_dataset,
            valid_sets=[
                train_dataset,
                val_dataset,
            ],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.log_evaluation(period=100),
            ],
            num_boost_round=2000,
        )

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