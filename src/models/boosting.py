import warnings
from collections import defaultdict

import lightgbm
import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sps
from hyperopt import fmin, hp, tpe
from loguru import logger
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from src.dataset import NBRDatasetBase
from src.models.core import IRecommender

warnings.filterwarnings("ignore")


def get_samples(m2time, user_id, only_last_basket=False):

    d = m2time.copy()
    rd = defaultdict(list)
    max_k = 0
    for k in d:
        for el in d[k]:
            rd[el].append(k)
            max_k = max(max_k, el)

    if only_last_basket:
        rd[max_k + 1] = [100000000000]

    samples = []

    sorted_basket_keys = sorted(rd.keys())

    consumed = set(rd[sorted_basket_keys[0]])

    for i in range(1, len(sorted_basket_keys)):

        not_consumed = consumed - set(rd[sorted_basket_keys[i]])

        for flag, ar in zip([1, 0], [rd[sorted_basket_keys[i]], list(not_consumed)]):

            if not i > len(sorted_basket_keys) - 20:
                break

            if only_last_basket:
                if i != len(sorted_basket_keys) - 1:
                    continue

            for item in ar:

                new_sample = {
                    "item_id": item,
                    "label": flag,
                    "basket_id": str(user_id) + "___" + str(i),
                }
                new_sample.update({f"prev_delta_{i}": 100000 for i in range(1, 10 + 1)})
                new_sample.update({f"delta_{t+1}_{t}": 100000 for t in range(9)})

                steps_cur = 0
                freq = 0
                for j in range(i - 1, -1, -1):
                    if item in rd[sorted_basket_keys[j]]:
                        if steps_cur < 9:
                            steps_cur += 1
                            new_sample[f"prev_delta_{steps_cur}"] = i - j

                        freq += 1

                deltas = []
                for t in range(9):
                    if new_sample[f"prev_delta_{t+2}"] == 100000:
                        break
                    new_sample[f"delta_{t+1}_{t}"] = new_sample[f"prev_delta_{t+2}"] - new_sample[f"prev_delta_{t+1}"]
                    deltas.append(new_sample[f"prev_delta_{t+2}"] - new_sample[f"prev_delta_{t+1}"])

                if len(deltas) == 0:
                    new_sample["delta_to_cur"] = 100000
                else:
                    new_sample["delta_to_cur"] = new_sample["prev_delta_1"] - np.mean(deltas)

                new_sample["freq"] = freq
                if freq:
                    samples.append(new_sample)

        for el in rd[sorted_basket_keys[i]]:
            consumed.add(el)

    return samples


def get_train_data_from_df(df, nrows=1000, max_window=8):

    sequences = []
    cur_user_id = df.user_id.values[0]
    m2time = defaultdict(list)
    nrows_ready = 0

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        new_user_id = row.user_id

        if new_user_id != cur_user_id:
            sequences.extend(get_samples(m2time, cur_user_id))

            m2time = defaultdict(list)

        merch_name = row.basket
        timestamps = [row.order_num] * len(merch_name)

        for x, y in zip(merch_name, timestamps):
            if len(m2time[x]) == 0 or (len(m2time[x]) and int(y) != m2time[x][-1]):
                m2time[x].append(int(y))

        cur_user_id = new_user_id
        nrows_ready += 1

        if nrows != -1 and nrows_ready > nrows:
            break

    sequences = pd.DataFrame(sequences)

    print(sequences.shape)

    print(sequences.shape)

    sequences.to_pickle("debug1.pkl")

    sequences = sequences.sample(frac=1)
    sequences["item_id"] = sequences["item_id"].astype("category")

    return sequences


def objective(hyperparameters: dict, n_splits, dataframes, features, verbose) -> float:
    hyperparameters["subsample"] /= 10
    hyperparameters["colsample_bytree"] /= 10

    metric = []

    train = dataframes

    model = lightgbm.LGBMRanker(n_jobs=10, objective="rank_xendcg", boosting_type="gbdt", **hyperparameters)

    cv_score = 0
    kf = KFold(n_splits=n_splits)
    for kf_train, kf_test in kf.split(train):

        q_train = train.iloc[kf_train].copy()
        q_test = train.iloc[kf_test].copy()

        qids_train = q_train.groupby("basket_id")["basket_id"].count().to_numpy()
        qids_test = q_test.groupby("basket_id")["basket_id"].count().to_numpy()

        model.fit(
            X=q_train[features],
            y=q_train["label"],
            eval_set=[(q_test[features], q_test["label"])],
            eval_group=[qids_test],
            verbose=verbose,
            group=qids_train,
            eval_at=10,
        )

        cv_score += model.best_score_["valid_0"]["ndcg@10"]

    metric.append(cv_score / n_splits)

    hyperparameters["metric"] = np.mean(metric)

    return -np.mean(metric)


def tune_params_and_fit(
    train: pd.DataFrame,
    full_train: pd.DataFrame,
    n_splits=4,
    verbose=100,
    num_params=10,
    features=None,
):
    """
    Подбираем гиперпараметры для бустинга по train-части через кросс-валидацию. Обучаем модель на трейне

    """

    lists_cases = {
        "min_child_weight": [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
        "reg_alpha": [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
        "reg_lambda": [0, 1e-1, 1, 5, 10, 20, 50, 100],
    }

    space = {
        "num_leaves": hp.randint("num_leaves", 6, 50),
        "min_child_samples": hp.randint("min_child_samples", 100, 500),
        "min_data_per_group": hp.randint("min_data_per_group", 10, 500),
        "n_estimators": hp.randint("n_estimators", 50, 1000),
        "subsample": hp.randint("subsample", 1, 10),
        "colsample_bytree": hp.randint("colsample_bytree", 1, 10),
    }
    for el in lists_cases.keys():
        space[el] = hp.choice(el, lists_cases[el])

    best = fmin(
        lambda x: objective(
            hyperparameters=x,
            n_splits=n_splits,
            dataframes=train,
            verbose=verbose,
            features=features,
        ),
        space,
        algo=tpe.suggest,
        max_evals=num_params,
    )

    logger.info(f"The Best set of params is found. They are: {str(best)}")

    best["subsample"] /= 10
    best["colsample_bytree"] /= 10

    for el in lists_cases.keys():
        best[el] = lists_cases[el][best[el]]

    final_model = lightgbm.LGBMRanker(n_jobs=10, objective="rank_xendcg", boosting_type="gbdt", **best)
    qids_train = train.groupby("basket_id")["basket_id"].count().to_numpy()

    final_model.fit(X=full_train[features], y=full_train["label"], group=qids_train)

    return None, final_model


def test_predictions(df, final_model, features, num_items=3000):

    cur_user_id = df.user_id.values[0]
    m2time = defaultdict(list)
    max_timestamp = 0
    user_vectors = np.zeros((df.user_id.max() + 1, num_items))

    for idx, row in tqdm(df.iterrows()):
        new_user_id = row.user_id

        if new_user_id != cur_user_id:

            samples = []

            samples.extend(get_samples(m2time, cur_user_id, only_last_basket=True))

            samples = pd.DataFrame(samples)

            samples["item_id"] = samples["item_id"].astype("category")

            samples["preds"] = final_model.predict(samples[features])
            for idx2, row2 in samples.iterrows():

                user_vectors[cur_user_id, int(row2.item_id)] = row2.preds

            m2time = defaultdict(list)
            max_timestamp = 0

        merch_name = row.basket
        timestamps = [row.order_num] * len(merch_name)

        for x, y in zip(merch_name, timestamps):
            if len(m2time[x]) == 0 or (len(m2time[x]) and int(y) != m2time[x][-1]):
                m2time[x].append(int(y))
                max_timestamp = max(max_timestamp, int(y))

        cur_user_id = new_user_id

    return user_vectors


def fit_ease(X, reg_weight=0.01):

    G = X.T @ X

    G += reg_weight * sps.identity(G.shape[0])

    G = G.todense()

    P = np.linalg.inv(G)
    B = P / (-np.diag(P))

    np.fill_diagonal(B, 0.0)

    return B


class OurRecommender(IRecommender):
    def __init__(
        self,
        num_nearest_neighbors: int = 300,
        within_decay_rate: float = 0.9,
        group_decay_rate: float = 0.7,
        alpha: float = 0.7,
        max_window: int = 7,
    ) -> None:
        super().__init__()
        self.num_nearest_neighbors = num_nearest_neighbors
        self.within_decay_rate = within_decay_rate
        self.group_decay_rate = group_decay_rate
        self.alpha = alpha
        self.max_window = max_window

        self._user_vectors = None
        self._nbrs = None

    def fit(self, dataset: NBRDatasetBase):

        FEATURES = ["freq", "item_id", "delta_to_cur"]

        FEATURES += [f"prev_delta_{i}" for i in range(1, self.max_window + 1)]

        FEATURES += [f"delta_{i+1}_{i}" for i in range(8)]

        fitted_model, final_model = tune_params_and_fit(
            train=dataset.sequences,
            full_train=dataset.sequences,
            n_splits=4,
            verbose=False,
            num_params=3,
            features=FEATURES,
        )
        print(FEATURES)
        print(final_model.feature_importances_)

        self._user_vectors = test_predictions(
            dataset.train_df,
            final_model,
            FEATURES,
            num_items=len(dataset._index2item),
        )

        self._user_vectors = sps.csr_matrix(self._user_vectors)

        self._nbrs = NearestNeighbors(
            n_neighbors=self.num_nearest_neighbors + 1,
            algorithm="brute",
        ).fit(self._user_vectors)

        return self

    def predict(self, user_ids, topk=None):
        if topk is None:
            topk = self._user_vectors.shape[1]

        user_vectors = self._user_vectors[user_ids, :]

        user_nn_indices = self._nbrs.kneighbors(user_vectors, return_distance=False)

        user_nn_vectors = []
        for nn_indices in user_nn_indices:
            nn_vectors = self._user_vectors[nn_indices[1:], :].mean(axis=0)
            user_nn_vectors.append(sps.csr_matrix(nn_vectors))
        user_nn_vectors = sps.vstack(user_nn_vectors)

        pred_matrix = self.alpha * user_vectors + (1 - self.alpha) * user_nn_vectors
        return pred_matrix

    @classmethod
    def sample_params(cls, trial: optuna.Trial) -> dict:
        num_nearest_neighbors = trial.suggest_categorical(
            "num_nearest_neighbors", [100, 300, 500, 700, 900, 1100, 1300]
        )
        within_decay_rate = trial.suggest_categorical(
            "within_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        )
        group_decay_rate = trial.suggest_categorical(
            "group_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        )
        alpha = trial.suggest_categorical("alpha", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        max_window = trial.suggest_int("max_window", 5, 10)

        return {
            "num_nearest_neighbors": num_nearest_neighbors,
            "within_decay_rate": within_decay_rate,
            "group_decay_rate": group_decay_rate,
            "alpha": alpha,
            "max_window": max_window,
        }
