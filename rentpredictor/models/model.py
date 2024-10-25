from abc import ABCMeta, abstractmethod
import pandas as pd

class Model(metaclass=ABCMeta):
    """모델에 대한 추상 클래스입니다.
    """

    @abstractmethod
    def set_data(self, dataframes: dict[str, pd.DataFrame]) -> None:
        """각종 데이터를 설정합니다.

        Parameters
        ----------
        dataframes : dict[str, pd.DataFrame]
            학습 및 테스트 데이터, 그리고 그 외 정보를 포함한 딕셔너리입니다.
        """

    @abstractmethod
    def preprocess(self) -> None:
        """데이터를 전처리합니다.
        """

    @abstractmethod
    def fit(self) -> None:
        """학습 데이터를 통해 모델을 학습합니다.
        """

    @abstractmethod
    def predict(self) -> pd.Series:
        """테스트 데이터에 대해 예측을 수행합니다.

        Returns
        -------
        pd.Series
            테스트 데이터에 대한 예측한 결과입니다.
        """

