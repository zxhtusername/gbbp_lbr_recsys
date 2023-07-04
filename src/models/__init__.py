# from .custom import OurRecommender
from .boosting import OurRecommender
from .core import IRecommender, IRecommenderNextTs

MODELS = {
    "boosting": OurRecommender,
}
