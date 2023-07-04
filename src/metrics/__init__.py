from .base import IMetric
from .ndcg import NDCG
from .recall import Recall

METRICS = {
    Recall.metric_name: Recall,
    NDCG.metric_name: NDCG,
}
