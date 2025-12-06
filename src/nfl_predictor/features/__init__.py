"""Feature engineering modules."""

from nfl_predictor.features.base_game_features import BaseGameFeatures
from nfl_predictor.features.feature_registry import register_feature_group

# Register all feature groups
register_feature_group(BaseGameFeatures())
