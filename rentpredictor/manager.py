import os
import importlib
import pandas as pd

from .preprocessor import Preprocessor
from .models.model import Model

class Manager:

    def __init__(self) -> None:
        self.preprocessor: Preprocessor = None
        self.model: Model = None

    def load_dataframes(self) -> None:
        """학습과 테스트에 사용될 모든 데이터를 로드합니다.
        """
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        self.train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
        self.test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
        self.park_df = pd.read_csv(os.path.join(data_path, 'parkInfo.csv'))
        self.school_df = pd.read_csv(os.path.join(data_path, 'schoolinfo.csv'))
        self.subway_df = pd.read_csv(os.path.join(data_path, 'subwayInfo.csv'))
        self.interest_df = pd.read_csv(os.path.join(data_path, 'interestRate.csv'))
        self.submission_df = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))


    def load_model(self, model_file_name: str) -> None:
        """모델의 클래스를 로드하고 생성합니다.

        Parameters
        ----------
        model_file_name : str
            로드할 모델의 파일명입니다.
        """
        
        if model_file_name.endswith('.py'):
            module_name = model_file_name[:-3]
        else:
            module_name = model_file_name
        model_name = ''.join([word.title() for word in module_name.split('_')])

        module = importlib.import_module('..models', package=__name__)
        model_cls = getattr(module, model_name)
        self.model = model_cls()

    def validate_model(self, train_start: int, train_end: int, val_start: int, val_end: int) -> tuple[pd.Series, pd.Series]:
        """모델을 학습하고 검증 데이터에 대한 예측값을 반환합니다.

        Parameters
        ----------
        train_start : int
            학습 데이터의 시작 연월입니다.
        train_end : int
            학습 데이터의 끝 연월입니다.
        val_start : int
            검증 데이터의 시작 연월입니다.
        val_end : int
            검증 데이터의 끝 연월입니다.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            검증 데이터의 실제값과 예측값입니다.
        """
        train_df = self.train_df[((self.train_df['contract_year_month'] >= train_start) &
                                  (self.train_df['contract_year_month'] <= train_end))].copy()
        val_df = self.train_df[((self.train_df['contract_year_month'] >= val_start) &
                                (self.train_df['contract_year_month'] <= val_end))].copy()
        val_y = val_df['deposit']
        val_df = val_df.drop(columns=['deposit'])
        self.preprocessor = Preprocessor(train_df, val_df, self.park_df, 
                                         self.school_df, self.subway_df, self.interest_df)
        self.model.preprocess(self.preprocessor)
        train_df = self.preprocessor.get_train_df()
        train_X, train_y = train_df.drop(columns=['deposit']), train_df['deposit']
        val_X = self.preprocessor.get_test_df()
        self.model.fit(train_X, train_y, val_X, val_y)
        val_y_pred = self.model.predict(val_X)
        return val_y, val_y_pred

    def test_model(self):
        """모델의 예측 결과를 파일로 저장합니다.
        """
        self.preprocessor = Preprocessor(self.train_df, self.test_df, self.park_df, 
                                         self.school_df, self.subway_df, self.interest_df)
        self.model.preprocess(self.preprocessor)
        train_df = self.preprocessor.get_train_df()
        train_X, train_y = train_df.drop(columns=['deposit']), train_df['deposit']
        test_X = self.preprocessor.get_test_df()
        
        self.model.fit(train_X, train_y)
        test_y_pred = self.model.predict(test_X)

        self.submission_df['deposit'] = test_y_pred
        self.submission_df.to_csv('output.csv', index=False)