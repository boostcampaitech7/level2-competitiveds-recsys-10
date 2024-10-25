import pandas as pd
from .models.model import Model
from . import utils

class Manager:

    def __init__(self):
        self.dataframes = utils.load_dataframes()

    def select_model(self, model_file_name: str) -> None:
        """모델의 클래스를 로드합니다.

        Parameters
        ----------
        model_file_name : str
            로드할 모델의 파일명입니다.
        hyperparams : dict, optional
            모델의 하이퍼파라미터입니다.
        """
        self.model_cls = utils.load_model_class(model_file_name)
    
    def validate_model(self, train_start: pd.Timestamp, train_end: pd.Timestamp,
                       val_start: pd.Timestamp, val_end: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
        """모델을 학습하고 검증 데이터에 대한 예측값을 반환합니다.

        Parameters
        ----------
        train_start : pd.Timestamp
            학습 데이터의 시작 연월입니다.
        train_end : pd.Timestamp
            학습 데이터의 끝 연월입니다.
        val_start : pd.Timestamp
            검증 데이터의 시작 연월입니다.
        val_end : pd.Timestamp
            검증 데이터의 끝 연월입니다.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            검증 데이터의 실제값과 예측값입니다.
        """
        train_start, train_end = int(train_start.strftime('%Y%m')), int(train_end.strftime('%Y%m'))
        val_start, val_end = int(val_start.strftime('%Y%m')), int(val_end.strftime('%Y%m'))
        all_df = self.dataframes['train_df']
        train_df = all_df[((all_df['contract_year_month'] >= train_start) &
                           (all_df['contract_year_month'] <= train_end))].copy()
        val_df = all_df[((all_df['contract_year_month'] >= val_start) &
                         (all_df['contract_year_month'] <= val_end))].copy()

        model: Model = self.model_cls()
        cur_dataframes = self.dataframes.copy()
        cur_dataframes['train_df'] = train_df
        cur_dataframes['test_df'] = val_df
        model.set_data(cur_dataframes)
        model.preprocess()
        model.fit()

        val_y = val_df['deposit'].copy()
        val_y_pred = model.predict()
        val_y.reset_index(drop=True, inplace=True)
        val_y_pred.reset_index(drop=True, inplace=True)
        return val_y, val_y_pred

    def test_model(self):
        """모델의 테스트 데이터에 대한 예측 결과를 반환합니다.
        """
        model: Model = self.model_cls()
        model.set_data(self.dataframes.copy())
        model.preprocess()
        model.fit()
        test_y_pred = model.predict()
        test_y_pred.reset_index(drop=True, inplace=True)
        return test_y_pred