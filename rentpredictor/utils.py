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

def _get_result_file_name(model_cls: Type[Model], train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    """결과 파일명을 반환합니다.

    Parameters
    ----------
    model_cls : Type[Model]
        모델의 클래스입니다.
    train_df : pd.DataFrame
        학습 데이터입니다.
    test_df : pd.DataFrame
        테스트 데이터입니다.
    
    Returns
    -------
    str
        결과 파일명입니다.
    """
    model_str = model_cls.__name__
    train_start, train_end = train_df['contract_year_month'].min(), train_df['contract_year_month'].max()
    test_start, test_end = test_df['contract_year_month'].min(), test_df['contract_year_month'].max()
    date_str = f'{train_start}-{train_end}_{test_start}-{test_end}'
    file_name = f'{model_str}_from_{date_str}.txt'
    return file_name

def load_result_if_exist(model_cls: Type[Model], train_df: pd.DataFrame, 
                         val_df: pd.DataFrame, result_path: str = None) -> pd.Series | None:
    """이전에 계산한 결과가 있다면 불러옵니다.
    
    Parameters
    ----------
    model_cls : Type[Model]
        모델의 클래스입니다.
    train_df : pd.DataFrame
        학습 데이터입니다.
    val_df : pd.DataFrame
        검증 데이터입니다.
    result_path : str, optional
        결과 파일이 저장된 경로입니다.
        전달되지 않으면, 패키지가 존재하는 디렉토리의 results 폴더에서 결과를 로드합니다.
    
    Returns
    -------
    pd.Series | None
        저장된 결과가 있다면 해당 결과를 반환하고, 그렇지 않다면 None을 반환합니다.
    """
    file_name = _get_result_file_name(model_cls, train_df, val_df)
    if result_path is None:
        result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    result_path = os.path.join(result_path, file_name)
    
    if os.path.exists(result_path):
        output = pd.read_csv(result_path, index_col=0)
        output = output['0']
    else:
        output = None
    return output

def save_result(output: pd.Series, model_cls: Type[Model], train_df: pd.DataFrame,
                val_df: pd.DataFrame, result_path: str = None) -> None:
    """결과를 저장합니다.

    Parameters
    ----------
    output : pd.Series
        저장할 결과입니다.
    model_cls : Type[Model]
        모델의 클래스입니다.
    train_df : pd.DataFrame
        학습 데이터입니다.
    val_df : pd.DataFrame
        검증 데이터입니다.
    result_path : str, optional
        결과 파일이 저장될 경로입니다.
        전달되지 않으면, 패키지가 존재하는 디렉토리의 results 폴더에 결과를 저장합니다.
    """
    file_name = _get_result_file_name(model_cls, train_df, val_df)
    if result_path is None:
        result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    result_path = os.path.join(result_path, file_name)
    
    output.to_csv(result_path, header=False)


    