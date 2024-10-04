from abc import ABCMeta, abstractmethod
import pandas as pd
from ..preprocessor import Preprocessor

class Model(metaclass=ABCMeta):
    """모델에 대한 추상 클래스입니다.
    """

    @abstractmethod
    def preprocess(self, preprocessor: Preprocessor) -> None:
        """preprocessor 객체를 이용하여 데이터를 전처리합니다.

        Parameters
        ----------
        preprocessor : Preprocessor
            전처리될 데이터를 담고 있는 Preprocessor 객체입니다.
        """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """모델을 학습합니다.

        Parameters
        ----------
        X : pd.DataFrame
            학습할 데이터입니다.
        y : pd.Series
            학습데이터에 대해 예측해야하는 target 값입니다.
        """

    @abstractmethod
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

