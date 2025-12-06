# NFL Predictor - Data Schemas

This document defines the schemas for all interim and processed data tables.

---

## Interim Tables (data/interim/)

These are cleaned, normalized tables that serve as inputs to feature engineering.

### 1. `games_base.parquet`

Core game-level table with results and metadata.

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | str | Unique game identifier (e.g., "2023_01_BUF_LAR") |
| `season` | int | Season year (e.g., 2023) |
| `week` | int | Week number (1-18 for regular season, 19+ for playoffs) |
| `game_date` | datetime | Date and time of kickoff |
| `game_type` | str | Game type: "REG", "WC", "DIV", "CONF", "SB" |
| `home_team` | str | Home team abbreviation (e.g., "BUF") |
| `away_team` | str | Away team abbreviation (e.g., "LAR") |
| `home_points` | int | Final score for home team |
| `away_points` | int | Final score for away team |
| `stadium_id` | str | Stadium identifier (nullable) |
| `kickoff_time_local` | datetime | Local kickoff time (nullable) |
| `neutral_site` | bool | Whether game is at neutral site (playoffs, international) |

**Derived Targets:**
| Column | Type | Description |
|--------|------|-------------|
| `home_win` | int | 1 if home team won, 0 otherwise |
| `margin` | int | Home points - away points |
| `total_points` | int | Home points + away points |

**Primary Key:** `game_id`

**Sort Order:** `game_date`, `game_id`

---

### 2. `stadiums.parquet`

Stadium metadata for venue-based features.

| Column | Type | Description |
|--------|------|-------------|
| `stadium_id` | str | Unique stadium identifier |
| `stadium_name` | str | Full stadium name |
| `primary_team` | str | Team that plays home games here (nullable) |
| `is_dome` | bool | Whether stadium is domed/retractable roof |
| `surface_type` | str | "grass", "turf", "fieldturf", etc. |
| `city` | str | City name |
| `state` | str | State abbreviation (US) or country (international) |
| `latitude` | float | Geographic latitude |
| `longitude` | float | Geographic longitude |
| `altitude_meters` | float | Altitude above sea level |
| `timezone` | str | IANA timezone (e.g., "America/New_York") |

**Primary Key:** `stadium_id`

**Usage:**
- Join to `games_base` on `stadium_id`
- Calculate travel distance from team home stadiums
- Altitude adjustments (e.g., Denver)

---

### 3. `weather_games.parquet`

Per-game weather conditions (when applicable).

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | str | Foreign key to games_base |
| `temperature_f` | float | Temperature in Fahrenheit (nullable for domes) |
| `feels_like_f` | float | Apparent temperature (nullable) |
| `wind_mph` | float | Wind speed in mph |
| `wind_gust_mph` | float | Wind gust speed (nullable) |
| `humidity` | float | Relative humidity (0-100) |
| `precip_type` | str | "none", "rain", "snow", "sleet" |
| `precip_intensity` | float | Precipitation intensity (mm/hr) |
| `visibility_miles` | float | Visibility in miles |
| `weather_source` | str | Data source (e.g., "darksky", "nflweather", "manual") |
| `measurement_time` | datetime | When weather was measured (typically at kickoff) |

**Primary Key:** `game_id`

**Notes:**
- For domed stadiums, weather fields may be null or external conditions
- Use `stadiums.is_dome` to determine if weather applies to gameplay

---

### 4. `betting_lines.parquet`

Vegas betting lines (if available).

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | str | Foreign key to games_base |
| `spread_opening` | float | Opening point spread (home team perspective) |
| `spread_closing` | float | Closing point spread (home team perspective) |
| `total_opening` | float | Opening over/under total points |
| `total_closing` | float | Closing over/under total points |
| `moneyline_home_open` | int | Opening home moneyline (American odds) |
| `moneyline_home_close` | int | Closing home moneyline |
| `moneyline_away_open` | int | Opening away moneyline |
| `moneyline_away_close` | int | Closing away moneyline |
| `book_source` | str | Sportsbook (e.g., "pinnacle", "fanduel", "consensus") |
| `line_movement` | float | Spread movement (closing - opening) |

**Primary Key:** `game_id`, `book_source`

**Derived Features:**
- Implied win probabilities from moneylines
- Vegas predicted margin = spread
- Vegas predicted total = total line
- Line movement as market signal

**Note:** These are **very strong** features. Always evaluate models both with and without betting lines.

---

### 5. `injuries.parquet`

Player injury reports (advanced feature).

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | str | Foreign key to games_base |
| `team` | str | Team abbreviation |
| `player_id` | str | Unique player identifier |
| `player_name` | str | Player name |
| `position` | str | Position abbreviation (QB, RB, WR, etc.) |
| `status` | str | Injury status: "out", "doubtful", "questionable", "IR" |
| `injury_type` | str | Body part/injury description (nullable) |
| `reported_date` | datetime | When injury was reported |
| `snap_share_prev_3` | float | Average snap share in previous 3 games (0-1) |
| `starter` | bool | Whether player is a starter |

**Primary Key:** `game_id`, `team`, `player_id`

**Aggregation Strategy:**
- Count starters out by position group (QB, OL, WR/TE, RB, DL, LB, DB)
- Weight by snap share or position importance
- Example features:
  - `starters_out_offense`, `starters_out_defense`
  - `qb_injured`, `oline_injuries_count`

---

### 6. `pbp.parquet` (Play-by-Play) - Advanced

Detailed play-by-play data for advanced metrics.

**Note:** This is a large table. See `nflverse/nfl_data_py` for schema reference.

**Key Columns:**
- `game_id`, `play_id`, `posteam`, `defteam`
- `down`, `ydstogo`, `yardline_100`
- `play_type`, `yards_gained`
- `epa`, `wpa`, `success`
- `passer_player_name`, `rusher_player_name`, `receiver_player_name`

**Usage:**
- Calculate advanced team stats: EPA/play, success rate, explosive play rate
- Generate drive-level features
- Create sequence inputs for RNN/Transformer models

---

## Processed Tables (data/processed/)

These are feature matrices ready for modeling.

### 7. `features_master.parquet`

Main feature matrix with all feature groups merged.

**Structure:**
| Column | Type | Description |
|--------|------|-------------|
| `game_id` | str | Game identifier |
| `game_date` | datetime | Game date (for time-based splits) |
| `season` | int | Season year |
| `week` | int | Week number |
| `home_team` | str | Home team |
| `away_team` | str | Away team |
| **[Base features]** | various | Week flags, game type, calendar features |
| **[Rolling stats]** | float | `home_pts_last3`, `away_pts_last3`, etc. |
| **[Elo]** | float | `home_elo_before`, `away_elo_before`, `elo_diff` |
| **[H2H]** | float | `h2h_games_played`, `h2h_win_pct_home`, etc. |
| **[Schedule]** | various | `is_divisional`, `rest_diff`, `travel_distance_away_km` |
| **[Venue]** | various | `is_dome`, `surface_type_encoded`, `altitude_high` |
| **[Weather]** | float | `temp_f_binned`, `wind_mph`, `has_precip` |
| **[Betting]** | float | `spread_closing`, `implied_home_win_prob` (if available) |
| **[Injuries]** | int | `starters_out_home_offense`, `qb_injured_away` (if available) |
| **[Sequence]** | float | `team_form_embed_home_0..15`, `team_form_embed_away_0..15` (advanced) |

**Targets:**
| Column | Type | Description |
|--------|------|-------------|
| `home_win` | int | Target for winner classification |
| `margin` | int | Target for margin regression |
| `total_points` | int | Target for total regression |

**Important Metadata:**
| Column | Type | Description |
|--------|------|-------------|
| `feature_max_date` | datetime | Max date used in computing features (for leakage checks) |
| `is_test` | bool | Whether game is in test set (for evaluation) |

**Constraints:**
- `feature_max_date < game_date` (enforced by validation)
- No nulls in critical features (impute or drop)

---

### 8. Feature Group Metadata

Each feature group can optionally export a metadata file:

**Example:** `data/processed/features_elo_metadata.json`

```json
{
  "feature_group": "elo",
  "version": "1.0",
  "features": ["home_elo_before", "away_elo_before", "elo_diff"],
  "config": {
    "k_factor": 20,
    "initial_elo": 1500,
    "reversion_factor": 0.33
  },
  "created_at": "2024-01-15T10:30:00Z",
  "leakage_checked": true
}
```

---

## Validation Rules

All interim and processed tables must pass these checks:

1. **Schema Compliance:**
   - All required columns present
   - Correct data types
   - Primary keys are unique

2. **Data Quality:**
   - No nulls in critical fields (game_id, teams, scores)
   - Dates are valid and in expected range
   - Foreign keys resolve (e.g., game_id in weather → game_id in games_base)

3. **Leakage Prevention:**
   - For features: `feature_max_date < game_date`
   - No use of target variables in feature computation
   - Time-based splits respect chronological order

4. **Consistency:**
   - All games in features_master exist in games_base
   - Feature value ranges are sensible (no extreme outliers without explanation)

**Validation Script:** `src/nfl_predictor/utils/validation.py`

---

## Schema Evolution

When schemas change:

1. **Version the change:** Add `_v2` suffix to table or column
2. **Document migration:** Update this file with migration notes
3. **Maintain backwards compatibility:** Keep old columns if possible
4. **Update tests:** Ensure validation tests cover new schema

**Example Migration:**
```markdown
## Migration: games_base v1 → v2 (2024-02-01)
- Added: `neutral_site` column (bool)
- Deprecated: `home_field_advantage` (derived from neutral_site + stadium_id)
- Backfilled: All historical games updated
```
