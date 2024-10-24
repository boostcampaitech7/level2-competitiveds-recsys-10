import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


class Preprocessor:
    """데이터 전처리를 수행하는 클래스입니다.
    """

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, park_df: pd.DataFrame,
                 school_df: pd.DataFrame, subway_df: pd.DataFrame, interest_df: pd.DataFrame) -> None:
        """전처리하는 데에 있어서 필요한 데이터들을 받고 저장합니다.

        Parameters
        ----------
        train_df : pd.DataFrame
            학습 데이터 프레임입니다.
        test_df : pd.DataFrame
            테스트 데이터 프레임입니다.
        park_df : pd.DataFrame
            공원 데이터 프레임입니다.
        school_df : pd.DataFrame
            학교 데이터 프레임입니다.
        subway_df : pd.DataFrame
            지하철 데이터 프레임입니다.
        interest_df : pd.DataFrame
            금리 데이터 프레임입니다.
        """
        self.train_df = train_df
        self.test_df = test_df
        self.park_df = park_df
        self.school_df = school_df
        self.subway_df = subway_df
        self.interest_df = interest_df

        # train_df와 test_df를 합친 DataFrame입니다.
        # 전처리는 이 DataFrame을 기반으로 진행됩니다.
        self.all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

        self.all_df['contract_type'] = self.all_df['contract_type'].astype('category')

        # (latitude, longitude) 튜플을 고유한 id로 변환하는 딕셔너리입니다.
        # 이는 위치 데이터를 다루는 데에 있어서 중복된 계산을 피하기 위함입니다.
        unique_locations = self.all_df[['latitude', 'longitude']].drop_duplicates().values.tolist()
        self.loc_to_id = {tuple(loc): id for id, loc in enumerate(unique_locations)}

        # (latitude, longitude, area_m2) 튜플을 고유한 id로 변환하는 딕셔너리입니다.
        unique_locations = self.all_df[['latitude', 'longitude', 'area_m2']].drop_duplicates().values.tolist()
        self.loc_area_to_id = {tuple(loc): id for id, loc in enumerate(unique_locations)}

    def remove_unnecessary_locations(self, epsilon: float):
        """예측하고자 하는 집들의 공간으로부터 epsilon 이상 떨어져있기에 예측에 있어서 불필요한 장소를 제거합니다.

        Parameters
        ----------
        epsilon : float
            제거할 장소를 결정하는 기준입니다.

            집들이 모여있는 사각형 경계로부터 epsilon 이상 떨어져있는 장소들을 제거합니다.
        """
        min_lat, max_lat = self.all_df['latitude'].min() - epsilon, self.all_df['latitude'].max() + epsilon
        min_long, max_long = self.all_df['longitude'].min() - epsilon, self.all_df['longitude'].max() + epsilon

        def get_clean_df(location_df: pd.DataFrame) -> pd.DataFrame:
            location_df = location_df[(location_df['latitude'] > min_lat) & (location_df['latitude'] < max_lat)]
            location_df = location_df[(location_df['longitude'] > min_long) & (location_df['longitude'] < max_long)]
            return location_df
        
        self.park_df = get_clean_df(self.park_df)
        self.school_df = get_clean_df(self.school_df)
        self.subway_df = get_clean_df(self.subway_df)

    def add_location_id(self) -> None:
        """위치에 따른 고유한 id를 추가합니다.
        """
        tuple_array = [self.all_df['latitude'].values, self.all_df['longitude'].values]
        self.all_df['location_id'] = pd.Series(
            [self.loc_to_id[(lat, long)] for lat, long in zip(*tuple_array)]
        ).astype('category')

    def add_location_with_area_id(self) -> None:
        """위치와 넓이에 따른 고유한 id를 추가합니다.
        """
        tuple_array = [self.all_df['latitude'].values, self.all_df['longitude'].values, self.all_df['area_m2'].values]
        self.all_df['location_with_area_id'] = pd.Series(
            [self.loc_area_to_id[(lat, long, area)] for lat, long, area in zip(*tuple_array)]
        ).astype('category')

    def add_contract_datetime(self) -> None:
        """계약 시간을 datetime 형식으로 변환합니다.
        """
        self.all_df['contract_datetime'] = (self.all_df['contract_year_month'].astype(str) + 
                                            self.all_df['contract_day'].astype(str))
        self.all_df['contract_datetime'] = pd.to_datetime(self.all_df['contract_datetime'], format='%Y%m%d')
    
    def add_latitude_bin(self, bin: float) -> None:
        """위도를 binning하여 category로 만든 열을 추가합니다.

        Parameters
        ----------
        bin : float
            bin의 크기입니다.
        """
        min_lat, max_lat = self.all_df['latitude'].min(), self.all_df['latitude'].max()
        self.all_df['latitude_bin'] = pd.cut(
            self.all_df['latitude'],                      
            bins=np.arange(min_lat, max_lat, bin)
        ).cat.codes.astype('category')

    def add_longitude_bin(self, bin: float) -> None:
        """경도를 binning하여 category로 만든 열을 추가합니다.

        Parameters
        ----------
        bin : float
            bin의 크기입니다.
        """
        min_long, max_long = self.all_df['longitude'].min(), self.all_df['longitude'].max()
        self.all_df['longitude_bin'] = pd.cut(
            self.all_df['longitude'],
            bins=np.arange(min_long, max_long, bin)
        ).cat.codes.astype('category')

    def add_min_distance_to_park(self, min_area: int = -1, max_area: int = 1e12) -> None:
        """가장 가까운 공원까지의 거리를 추가합니다.

        add_location_id를 먼저 호출해야합니다.

        Parameters
        ----------
        min_area : int, optional
            공원의 최소 면적입니다. 이보다 작은 공원은 고려하지 않습니다.
        max_area : int, optional
            공원의 최대 면적입니다. 이보다 큰 공원은 고려하지 않습니다.
        """
        park_df = self.park_df[(self.park_df['area'] >= min_area) & (self.park_df['area'] <= max_area)]
        park_tree = KDTree(park_df[['latitude', 'longitude']])

        id_to_distance = {id: park_tree.query([loc])[0][0] for loc, id in self.loc_to_id.items()}
        feature_name = 'min_distance_to_park'
        if min_area > -1:
            feature_name += f'_min_{min_area}'
        if max_area < 1e12:
            feature_name += f'_max_{max_area}'
        self.all_df[feature_name] = self.all_df['location_id'].map(id_to_distance)

    def add_min_distance_to_school(self, school_type: str = 'all') -> None:
        """가장 가까운 학교까지의 거리를 추가합니다.

        add_location_id를 먼저 호출해야합니다.

        Parameters
        ----------
        school_type : str, optional
            고려할 학교의 종류입니다. 다음 중 하나일 수 있습니다.

            'elementary': 초등학교
            'middle': 중학교
            'high': 고등학교
            'all': 모든 종류의 학교
        """
        school_df = self.school_df.copy()
        if school_type != 'all':
            school_df = school_df[school_df['schoolLevel'] == school_type]
        school_tree = KDTree(school_df[['latitude', 'longitude']])

        id_to_distance = {id: school_tree.query([loc])[0][0] for loc, id in self.loc_to_id.items()}
        feature_name = f'min_distance_to_school_{school_type}'
        self.all_df[feature_name] = self.all_df['location_id'].map(id_to_distance)

    def add_min_distance_to_subway(self) -> None:
        """가장 가까운 지하철까지의 거리를 추가합니다.

        add_location_id를 먼저 호출해야합니다.
        """
        subway_tree = KDTree(self.subway_df[['latitude', 'longitude']])

        id_to_distance = {id: subway_tree.query([loc])[0][0] for loc, id in self.loc_to_id.items()}
        self.all_df['min_distance_to_subway'] = self.all_df['location_id'].map(id_to_distance)

    def add_locations_within_radius(self, location_type: str, radius: float) -> None:
        """반경 내에 있는 특정 장소의 수를 추가합니다.

        add_location_id를 먼저 호출해야합니다.

        Parameters
        ----------
        location_type : str
            장소의 종류입니다. 'subway', 'school', 'park' 중 하나일 수 있습니다.
        radius : float
            반경입니다.
        """
        loc_df = getattr(self, f'{location_type}_df')
        loc_tree = KDTree(loc_df[['latitude', 'longitude']])
        id_to_loc_nums = {id: loc_tree.query_ball_point(loc, r=radius, return_length=True)
                          for loc, id in self.loc_to_id.items()}
        feature_name = f'num_{location_type}_within_{radius}'
        self.all_df[feature_name] = self.all_df['location_id'].map(id_to_loc_nums)

    def add_interest_rate(self) -> None:
        """금리 데이터를 추가합니다.

        contract_year_month 행이 존재해야합니다.
        """
        self.all_df = self.all_df.merge(self.interest_df,
                                        left_on='contract_year_month',
                                        right_on='year_month', 
                                        how='left')

    def drop_columns(self, column_names: list[str]) -> None:
        """불필요한 열을 제거합니다.

        Parameters
        ----------
        column_names : list[str]
            제거핧 열의 이름입니다.
        """
        self.all_df = self.all_df.drop(columns=column_names)

    def find_optimal_clusters(self, features: np.ndarray, max_k: int, min_k: int, sample_size: float = 0.1, random_state=42) -> int:
        """
        엘보우 방법과 실루엣 스코어, davies_bouldin_score를 사용하여 최적의 클러스터 개수를 찾습니다.

        Parameters
        ----------
        features : np.ndarray
            클러스터링에 사용할 특징 데이터입니다..
        max_k : int, optional
            클러스터의 최대 개수입니다.
        min_k : int, optional
            클러스터의 최소 개수입니다.
        sample_size : float, optional
            샘플링 비율입니다. 기본값은 10%입니다.

        Returns
        -------
        int
            최적의 클러스터 개수입니다.
        """
        np.random.seed(random_state)
        sample_indices = np.random.choice(features.shape[0], int(features.shape[0] * sample_size), replace=False)
        sampled_features = features[sample_indices]

        inertia = []
        silhouette_scores = []
        davies_bouldin_scores = []

        for k in range(min_k, max_k + 1):
            print(f"Clustering with k = {k} started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            start_time = time.time()

            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(sampled_features)
            inertia.append(kmeans.inertia_)

            silhouette_score_value = silhouette_score(sampled_features, kmeans.labels_)
            silhouette_scores.append(silhouette_score_value)

            davies_bouldin_score_value = davies_bouldin_score(sampled_features, kmeans.labels_)
            davies_bouldin_scores.append(davies_bouldin_score_value)

            end_time = time.time()
            print(f"Clustering with k = {k} finished at {time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {end_time - start_time:.2f} seconds)")

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(range(min_k, max_k + 1), inertia, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')

        plt.subplot(1, 3, 2)
        plt.plot(range(min_k, max_k + 1), silhouette_scores, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for Optimal k')

        plt.subplot(1, 3, 3)
        plt.plot(range(min_k, max_k + 1), davies_bouldin_scores, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Davies-Bouldin Index')
        plt.title('Davies-Bouldin Index for Optimal k')

        plt.tight_layout()
        plt.show()


        optimal_k = range(min_k, max_k + 1)[silhouette_scores.index(max(silhouette_scores))]
        print(f"Optimal k found: {optimal_k}")
        return optimal_k

    def add_location_based_cluster(self, train_df, test_df, max_k: int = 500, min_k: int = 450, sample_size: float = 0.1, random_state=42) -> int:
        """
        훈련 데이터의 위치(위도와 경도)를 기반으로 최적의 클러스터 ID를 찾고,
        해당 결과를 바탕으로 테스트 데이터에 클러스터를 할당합니다.

        Parameters
        ----------
        train_df : pd.DataFrame
            훈련 데이터.
        test_df : pd.DataFrame
            테스트 데이터.
        max_k : int, optional
            클러스터의 최대 개수입니다.
        min_k : int, optional
            클러스터의 최소 개수입니다.
        sample_size : float, optional
            샘플링 비율입니다. 기본값은 10%입니다.
        random_state : int, optional
            랜덤 시드입니다. 기본값은 42입니다.

        Returns
        -------
        int
            최적의 클러스터 개수입니다.
        """
        np.random.seed(random_state)
        train_location_features = train_df[['location_id']]

        scaler = StandardScaler()
        scaled_train_location_features = scaler.fit_transform(train_location_features)

        optimal_k = self.find_optimal_clusters(scaled_train_location_features, max_k,   min_k, sample_size)

        kmeans = KMeans(n_clusters=optimal_k, random_state=random_state)
        kmeans.fit(scaled_train_location_features)

        train_df['location_cluster'] = pd.Categorical(kmeans.labels_)
        test_location_features = test_df[['location_id']]
        scaled_test_location_features = scaler.transform(test_location_features)
        test_df['location_cluster'] = pd.Categorical(kmeans.predict (scaled_test_location_features))

        self.train_df = train_df
        self.test_df = test_df

        return optimal_k
    
    def add_min_distance_to_transfer_station(self) -> None:
        """
        각 아파트에서 가장 가까운 환승역까지의 거리를 추가합니다.
        환승역은 위도와 경도가 동일한 역들로 정의됩니다.
        """
        transfer_station_groups = self.subway_df.groupby(['latitude', 'longitude']).size().reset_index(name='transfer_lines_count')
        transfer_df = transfer_station_groups[transfer_station_groups['transfer_lines_count'] >= 2]
        transfer_tree = KDTree(transfer_df[['latitude', 'longitude']])
        id_to_distance = {id: transfer_tree.query([loc])[0][0] for loc, id in self.loc_to_id.items()}
        self.all_df['min_distance_to_transfer_station'] = self.all_df['location_id'].map(id_to_distance)

    def add_transfer_stations_within_radius(self, radius: float) -> None:
        """
        반경 내에 있는 환승역의 수를 추가합니다.
        환승역은 위도와 경도가 동일한 역들로 정의됩니다.

        Parameters
        ----------
        radius : float
            반경 내 환승역을 계산할 반경 (km 단위).
        """
        transfer_station_groups = self.subway_df.groupby(['latitude', 'longitude']).size().reset_index(name='transfer_lines_count')
        transfer_df = transfer_station_groups[transfer_station_groups['transfer_lines_count'] >= 2]
        transfer_tree = KDTree(transfer_df[['latitude', 'longitude']])
        id_to_transfer_count = {id: len(transfer_tree.query_ball_point(loc, r=radius)) for loc, id in self.loc_to_id.items()}
        self.all_df[f'num_transfer_stations_within_{radius}'] = self.all_df['location_id'].map(id_to_transfer_count)

    def get_train_df(self) -> pd.DataFrame:
        """전처리된 학습 데이터 프레임을 반환합니다.

        Returns
        -------
        pd.DataFrame
            전처리된 학습 데이터 프레임입니다.
        """
        train_df = self.all_df.iloc[:len(self.train_df)]
        return train_df

    def get_test_df(self) -> pd.DataFrame:
        """전처리된 테스트 데이터 프레임을 반환합니다.

        Returns
        -------
        pd.DataFrame
            전처리된 테스트 데이터 프레임입니다.
            deposit 열은 제거되어 있습니다.
        """
        test_df = self.all_df.iloc[len(self.train_df):]
        test_df = test_df.drop(columns='deposit')
        return test_df

