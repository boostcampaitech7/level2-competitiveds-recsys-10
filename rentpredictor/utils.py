import os
import importlib
from typing import Type

import numpy as np
import pandas as pd
from .models.model import Model

def load_dataframes(data_path: str = None) -> dict[str, pd.DataFrame]:
    """학습과 테스트에 사용될 모든 데이터를 로드합니다.

    Parameters
    ----------
    data_path : str, optional
        데이터가 저장된 경로입니다.
        전달되지 않으면, 패키지가 존재하는 디렉토리의 data 폴더에서 데이터를 로드합니다.
    
    Returns
    -------
    dict[str, pd.DataFrame]
        데이터프레임들이 저장된 딕셔너리입니다
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'), index_col=0)
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'), index_col=0)
    test_df['deposit'] = np.nan
    park_df = pd.read_csv(os.path.join(data_path, 'parkInfo.csv'))
    school_df = pd.read_csv(os.path.join(data_path, 'schoolinfo.csv'))
    subway_df = pd.read_csv(os.path.join(data_path, 'subwayInfo.csv'))
    interest_df = pd.read_csv(os.path.join(data_path, 'interestRate.csv'))
    dataframes = {
        'train_df': train_df,
        'test_df': test_df,
        'park_df': park_df,
        'school_df': school_df,
        'subway_df': subway_df,
        'interest_df': interest_df
    }
    return dataframes

def load_model_class(model_file_name: str) -> Type[Model]:
    """모델의 클래스를 로드합니다.

    Parameters
    ----------
    model_file_name : str
        로드할 모델의 파일명입니다.

    Returns
    -------
    Type[Model]
        로드된 모델의 클래스입니다.
    """ 
    module_name = model_file_name.removesuffix('.py')
    model_name = ''.join([word.title() for word in module_name.split('_')])
    module = importlib.import_module('..models', package=__name__)
    model_cls = getattr(module, model_name)
    return model_cls
