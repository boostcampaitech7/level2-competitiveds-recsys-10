import os
import importlib
import pandas as pd

from .preprocessor import Preprocessor
from .models.model import Model

class Manager:

    def __init__(self) -> None:
        self.preprocessor: Preprocessor = None
        self.model: Model = None
        self.prediction: pd.Series = None

    def run(self, model_file_name: str) -> None:
        """모델을 실행하고 그 결과를 저장합니다.

        Parameters
        ----------
        model_file_name : str
            모델 클래스의 파일명입니다.
        """
        print('데이터를 로드합니다.')
        self._load_dataframes()
        print('모델을 로드합니다.')
        self._load_model(model_file_name)
        print('모델을 실행합니다.')
        self._process_model()
        print('테스트 데이터에 대한 예측 결과를 저장합니다.')
        self._dump_model_output()

    def _load_dataframes(self) -> None:
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


    def _load_model(self, model_file_name: str) -> None:
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

    def _process_model(self):
        """모델을 학습하고 테스트 데이터에 대한 예측을 수행합니다.
        """
        self.preprocessor = Preprocessor(self.train_df, self.test_df, self.park_df, 
                                         self.school_df, self.subway_df, self.interest_df)
        self.model.preprocess(self.preprocessor)
        train_df = self.preprocessor.get_train_df()
        train_X, train_y = train_df.drop(columns=['deposit']), train_df['deposit']
        test_X = self.preprocessor.get_test_df()
        
        self.model.fit(train_X, train_y)
        self.prediction = self.model.predict(test_X)

    def _dump_model_output(self):
        """모델의 예측 결과를 파일로 저장합니다.
        """
        self.submission_df['deposit'] = self.prediction
        self.submission_df.to_csv('output.csv', index=False)