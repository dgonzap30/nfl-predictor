"""Data schemas for NFL predictor tables."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class GameBase:
    """Schema for games_base table."""

    game_id: str
    season: int
    week: int
    game_date: datetime
    game_type: str  # REG / POST / WC / DIV / CONF / SB
    home_team: str
    away_team: str
    home_points: int
    away_points: int
    stadium_id: Optional[str] = None
    kickoff_time_local: Optional[datetime] = None
    neutral_site: bool = False

    # Derived targets
    @property
    def home_win(self) -> int:
        """1 if home team won, 0 otherwise."""
        return 1 if self.home_points > self.away_points else 0

    @property
    def margin(self) -> int:
        """Home points - away points."""
        return self.home_points - self.away_points

    @property
    def total_points(self) -> int:
        """Total points scored in game."""
        return self.home_points + self.away_points


@dataclass
class Stadium:
    """Schema for stadiums table."""

    stadium_id: str
    stadium_name: str
    primary_team: Optional[str]
    is_dome: bool
    surface_type: str  # grass, turf, fieldturf, etc.
    city: str
    state: str  # US state or country for international
    latitude: float
    longitude: float
    altitude_meters: float
    timezone: str  # IANA timezone


@dataclass
class WeatherGame:
    """Schema for weather_games table."""

    game_id: str
    temperature_f: Optional[float]
    feels_like_f: Optional[float]
    wind_mph: float
    wind_gust_mph: Optional[float]
    humidity: float
    precip_type: str  # none, rain, snow, sleet
    precip_intensity: float
    visibility_miles: Optional[float]
    weather_source: str
    measurement_time: datetime


@dataclass
class BettingLine:
    """Schema for betting_lines table."""

    game_id: str
    spread_opening: Optional[float]
    spread_closing: float
    total_opening: Optional[float]
    total_closing: float
    moneyline_home_open: Optional[int]
    moneyline_home_close: int
    moneyline_away_open: Optional[int]
    moneyline_away_close: int
    book_source: str
    line_movement: Optional[float] = None


@dataclass
class Injury:
    """Schema for injuries table."""

    game_id: str
    team: str
    player_id: str
    player_name: str
    position: str
    status: str  # out, doubtful, questionable, IR
    injury_type: Optional[str]
    reported_date: datetime
    snap_share_prev_3: Optional[float]
    starter: bool
