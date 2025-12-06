"""Central registry of feature groups."""

from typing import Any, Dict, Protocol

import pandas as pd


class FeatureGroup(Protocol):
    """Protocol for feature group builders.

    Each feature group must implement this interface to be registered
    in the feature registry.
    """

    name: str

    def build_features(
        self,
        games_base: pd.DataFrame,
        aux_tables: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Build features for this group.

        Args:
            games_base: DataFrame with base game information
            aux_tables: Dictionary of auxiliary tables (stadiums, weather, etc.)
            config: Configuration dictionary for this feature group

        Returns:
            DataFrame with game_id and computed features

        Raises:
            ValueError: If data leakage is detected or required data is missing
        """
        ...


FEATURE_GROUPS: Dict[str, FeatureGroup] = {}


def register_feature_group(group: FeatureGroup) -> None:
    """Register a feature group in the global registry.

    Args:
        group: Feature group instance implementing FeatureGroup protocol

    Raises:
        ValueError: If a group with the same name is already registered
    """
    if group.name in FEATURE_GROUPS:
        raise ValueError(f"Feature group '{group.name}' is already registered")
    FEATURE_GROUPS[group.name] = group


def build_all_features(
    games_base: pd.DataFrame,
    aux_tables: Dict[str, pd.DataFrame],
    feature_group_names: list[str],
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Build features from multiple feature groups and merge them.

    Args:
        games_base: DataFrame with base game information
        aux_tables: Dictionary of auxiliary tables
        feature_group_names: List of feature group names to include
        config: Configuration dictionary (may contain group-specific configs)

    Returns:
        DataFrame with game_id and all features merged

    Raises:
        ValueError: If any feature group is not registered
    """
    if not feature_group_names:
        raise ValueError("Must specify at least one feature group")

    # Start with games_base as the base
    result = games_base[["game_id", "game_date", "season", "week"]].copy()

    # Build and merge each feature group
    for group_name in feature_group_names:
        if group_name not in FEATURE_GROUPS:
            raise ValueError(f"Feature group '{group_name}' is not registered")

        group = FEATURE_GROUPS[group_name]
        group_config = config.get(group_name, {})

        features = group.build_features(games_base, aux_tables, group_config)
        result = result.merge(features, on="game_id", how="left")

    return result


def build_features_v0(games_base: pd.DataFrame) -> pd.DataFrame:
    """Build the V0 feature matrix with all V0 feature groups.

    This is a convenience function that builds all V0 baseline features:
    - base_game_v0: Targets and calendar features
    - team_rolling_v0: 3-game rolling statistics
    - elo_v0: Elo ratings

    Args:
        games_base: DataFrame with base game information

    Returns:
        DataFrame with game_id, context columns, and all V0 features
    """
    return build_all_features(
        games_base=games_base,
        aux_tables={},
        feature_group_names=["base_game_v0", "team_rolling_v0", "elo_v0"],
        config={},
    )
