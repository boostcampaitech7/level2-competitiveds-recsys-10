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
        preprocessor.add_locations_within_radius('school', 0.1)

        # 위도, 경도를 binning하여 추가합니다.
        preprocessor.add_latitude_bin(0.01)
        preprocessor.add_longitude_bin(0.01)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """모델을 학습합니다.

        Parameters
        ----------
        X : pd.DataFrame
            학습할 데이터입니다.
        y : pd.Series
            학습데이터에 대해 예측해야하는 target 값입니다.
        """
        self.lgb_model = lgb.train(
            params={
                'objective': 'regression',
                'num_leaves': 63,
                'seed': 42,
                'verbose': -1,
            },
            train_set=lgb.Dataset(X, y),
            num_boost_round=2000,
        )

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
        y_pred = self.lgb_model.predict(X)
        return pd.Series(y_pred)