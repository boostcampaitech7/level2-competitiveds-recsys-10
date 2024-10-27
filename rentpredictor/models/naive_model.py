import numpy as np
import pandas as pd
from tqdm import tqdm
from .model import Model
from ..preprocessor import Preprocessor

class NaiveModel(Model):
    """이전 거래들의 지수 가중 평균을 예측으로 두는 모델입니다.
    """
    def set_data(self, dataframes: dict[str, pd.DataFrame]) -> None:
        """학습 및 테스트 데이터를 설정합니다.

        Parameters
        ----------
        dataframes : dict[str, pd.DataFrame]
            학습 및 테스트 데이터를 포함한 딕셔너리입니다.
        """
        self.dataframes = dataframes

    def preprocess(self) -> None:
        """id와 거래 시간 관련 feature를 추가합니다.
        """
        preprocessor = Preprocessor(self.dataframes)
        preprocessor.add_location_id()
        preprocessor.add_location_with_area_id()
        preprocessor.add_contract_datetime()
        self.train_df = preprocessor.get_train_df()
        self.test_df = preprocessor.get_test_df()

    def fit(self) -> None:
        """모델을 학습합니다.
        """
        house_df = self.train_df.copy()
        house_df = house_df[['location_with_area_id', 'deposit', 'contract_datetime']]
        house_df.sort_values(by=['location_with_area_id', 'contract_datetime'], inplace=True)
        grouped_house_df = house_df.groupby('location_with_area_id', observed=True)

        self.area_id_to_deposit_pred = {
            loc_area_id: group['deposit'].ewm(alpha=0.5).mean().iloc[-1]
            for loc_area_id, group in grouped_house_df
        }

    def predict(self) -> pd.Series:
        """테스트 데이터에 대해 예측을 수행합니다.

        Returns
        -------
        pd.Series
            테스트데이터에 대한 예측한 결과입니다.
        """
        y_pred = self.test_df['location_with_area_id'].map(self.area_id_to_deposit_pred)
        return y_pred