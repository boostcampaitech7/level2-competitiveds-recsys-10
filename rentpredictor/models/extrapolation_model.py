import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

from .model import Model
from ..preprocessor import Preprocessor


class ExtrapolationModel(Model):
    """모델에 대한 추상 클래스입니다.
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
        preprocessor.add_location_id()
        preprocessor.add_location_with_area_id()
        preprocessor.add_contract_datetime()
        self.train_df = preprocessor.get_train_df()
        self.test_df = preprocessor.get_test_df()

        # 아파트 별로 이동평균 그래프를 만듭니다.
        start_datetime = self.train_df['contract_datetime'].min()
        end_datetime = self.train_df['contract_datetime'].max()
        self.train_df.sort_values(by=['location_with_area_id', 'contract_datetime'], inplace=True)
        grouped_deposit_series = self.train_df.groupby('location_with_area_id', observed=True)
        full_date_range = pd.date_range(start=start_datetime, end=end_datetime, freq='1D')

        area_id_to_deposit_series = {
            loc_area_id: deposit_series.set_index('contract_datetime')['deposit']
                                    .ewm(alpha=0.5).mean()
                                    .resample('1d').last()
                                    .reindex(full_date_range)
                                    .interpolate(method='linear', limit_area='inside')
            for loc_area_id, deposit_series in grouped_deposit_series
        }

        # 모델이 학습할 target 데이터를 만듭니다.            
        time_span = pd.Timedelta(days=360)
        delta = pd.Timedelta(days=30)
        target_list = []
        target_date_range = pd.date_range(start=end_datetime, end=start_datetime + time_span, freq=-delta)
        for cur_timestamp_id, cur_datetime in enumerate(target_date_range):
            prev_datetime = cur_datetime - time_span
            for area_loc_id, deposit_series in area_id_to_deposit_series.items():
                prev_deposit = deposit_series[prev_datetime]
                cur_deposit = deposit_series[cur_datetime]
                deposit_pct_change = (cur_deposit - prev_deposit) / prev_deposit
                if not pd.isna(deposit_pct_change):
                    target_list.append((area_loc_id, cur_timestamp_id, deposit_pct_change))
        
        self.area_id_to_deposit_series = area_id_to_deposit_series
        self.target_list = target_list
        self.house_num = self.train_df['location_with_area_id'].nunique()
        self.time_num = len(target_date_range)

    def fit(self) -> None:
        """학습 데이터를 통해 모델을 학습합니다.
        """
        config = {
            'epoch': 3,
            'learning_rate': 1e-5,
            'house_embedding_dim': 16,
            'time_embedding_dim': 64,
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EmbeddingInteraction(self.house_num, self.time_num, config).to(self.device)
        self.model.house_embedding = self.model.house_embedding.to(self.device)
        self.model.time_embedding = self.model.time_embedding.to(self.device)
        self.model.interaction = self.model.interaction.to(self.device)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config['learning_rate']
        )

        house_ids, time_ids, targets = zip(*self.target_list)
        house_ids_tensor = torch.tensor(house_ids, dtype=torch.int).to(self.device)
        time_ids_tensor = torch.tensor(time_ids, dtype=torch.int).to(self.device)
        targets_tensor = torch.tensor(targets, dtype=torch.float).to(self.device)

        dataset = TensorDataset(house_ids_tensor, time_ids_tensor, targets_tensor)
        dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)

        for epoch in range(1, config['epoch'] + 1):
            for i, (house_ids_tensor, time_ids_tensor, targets_tensor) in enumerate(dataloader):
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                predictions = self.model.get_predictions(house_ids_tensor, time_ids_tensor)
                loss = torch.sum((predictions - targets_tensor) ** 2) / len(targets_tensor)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    pred = self.predict()
                    true = self.test_df['deposit']
                    pred, true = pred.reset_index(drop=True), true.reset_index(drop=True)
                    pred, true = pred[pred.notna()], true[pred.notna()]
                    mae = mean_absolute_error(true, pred)
                    print(f'Epoch {epoch}, MAE: {mae}')

    def predict(self) -> pd.Series:
        """테스트 데이터에 대해 예측을 수행합니다.

        Returns
        -------
        pd.Series
            테스트 데이터에 대한 예측한 결과입니다.
        """

        full_area_ids = torch.arange(self.model.house_num).to(self.device)
        last_time_ids = torch.tensor([1] * self.model.house_num).to(self.device)
        predictions = self.model.get_predictions(full_area_ids, last_time_ids)

        area_id_to_pred_deposit = {}
        prediction_date = self.train_df['contract_datetime'].max() - pd.Timedelta(days=30)
        previous_date = prediction_date - pd.Timedelta(days=360)
        for area_id, deposit_series in self.area_id_to_deposit_series.items():
            last_valid_date = deposit_series.last_valid_index()
            if previous_date <= last_valid_date <= prediction_date - pd.Timedelta(days=270):
                prev_deposit = deposit_series[previous_date]
                pct_change = predictions[area_id].item()
                next_deposit = prev_deposit * (1 + pct_change)
                area_id_to_pred_deposit[area_id] = float(next_deposit)
        pred = self.test_df['location_with_area_id'].map(area_id_to_pred_deposit)
        return pred

class EmbeddingInteraction(torch.nn.Module):
    def __init__(self, house_num: int, time_num: int, config: dict) -> None:
        super().__init__()
        self.house_num = house_num
        self.time_num = time_num
        self.config = config

        self.house_embedding = torch.nn.Embedding(self.house_num, self.config['house_embedding_dim'])
        self.time_embedding = torch.nn.Embedding(self.time_num, self.config['time_embedding_dim'])

        all_dim = self.config['house_embedding_dim'] + self.config['time_embedding_dim']
        self.interaction = torch.nn.Sequential(
            torch.nn.Linear(all_dim, all_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(all_dim // 2, all_dim // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(all_dim // 4, 1),
        )

    def get_predictions(self, house_ids: torch.Tensor, time_ids: torch.Tensor) -> torch.Tensor:
        house_embedding = self.house_embedding(house_ids)
        time_embedding = self.time_embedding(time_ids)
        all_embeddings = torch.cat([house_embedding, time_embedding], dim=1)
        all_scores = self.interaction(all_embeddings)
        return all_scores