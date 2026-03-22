from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator
from pathlib import Path


# ==========================================================
# GLOBAL SCIENTIFIC CONTEXT
# ==========================================================
#
# This configuration schema defines all parameters controlling
# the empirical bovine thermal comfort pipeline in psychrometric space.
#
# ----------------------------------------------------------
# PIPELINE OVERVIEW
# ----------------------------------------------------------
#
# The workflow controlled by this configuration is:
#
#   raw dataset
#       -> comfort filtering
#       -> psychrometric density field
#       -> optional density filtering
#       -> optional clustering
#       -> geometric polygon extraction
#       -> optional smoothing
#       -> multi-zone segmentation
#
# ----------------------------------------------------------
# IMPORTANT PRINCIPLE
# ----------------------------------------------------------
#
# These parameters are NOT universal biological constants.
#
# They are:
#   → methodological choices
#   → dataset-dependent
#   → tunable
#
# Reproducibility is achieved by:
#   → modifying ONLY these parameters
#   → keeping pipeline logic unchanged
#
# ==========================================================


# ==========================================================
# DENSITY CONFIGURATION
# ==========================================================
#
# Controls the construction of the 2D density field in
# psychrometric space:
#
#   x-axis → temperature (T)
#   y-axis → humidity ratio (W)
#
# Each bin represents the probability mass of comfort records.
#
# ----------------------------------------------------------
# SCIENTIFIC ROLE
# ----------------------------------------------------------
#
# Defines how the empirical distribution of comfort is represented.
#
# Impacts:
#   - spatial resolution
#   - noise level
#   - geometric stability
#
class DensityConfig(BaseModel):
    bins: int = 40
    min_density: float = 0.001
    percentile: float = 65
    use_filter: bool = False

    # ------------------------------------------------------
    # VALIDATIONS
    # ------------------------------------------------------
    @field_validator("bins")
    @classmethod
    def validate_bins(cls, v):
        if v < 10:
            raise ValueError("bins must be >= 10 for meaningful resolution")
        return v

    @field_validator("min_density")
    @classmethod
    def validate_min_density(cls, v):
        if v < 0:
            raise ValueError("min_density must be non-negative")
        return v

    @field_validator("percentile")
    @classmethod
    def validate_percentile(cls, v):
        if not (0 <= v <= 100):
            raise ValueError("percentile must be between 0 and 100")
        return v


# ==========================================================
# DENSITY FIELD INTERPRETATION (EXTENDED DOC)
# ==========================================================
#
# bins:
#   Controls histogram resolution
#
#   30 → smoother
#   40 → balanced (recommended baseline)
#   50 → more detailed (risk: fragmentation)
#
# min_density:
#   Removes weakly supported bins
#
#   High → aggressive cleaning
#   Low → preserves noise
#
# use_filter + percentile:
#   Percentile-based filtering of density field
#
#   50 → broad comfort
#   65 → balanced envelope
#   80 → core comfort
#
#   High percentile → smaller region
#   Low percentile → larger region
#
# ==========================================================


# ==========================================================
# CLUSTERING CONFIGURATION
# ==========================================================
#
# Used to isolate the main comfort region in psychrometric space.
#
# ----------------------------------------------------------
# SCIENTIFIC ROLE
# ----------------------------------------------------------
#
# Removes:
#   - satellite regions
#   - noise clusters
#
# Keeps:
#   - dominant comfort structure
#
# Uses DBSCAN because:
#   - no need to predefine number of clusters
#   - robust to irregular shapes
#   - detects noise naturally
#
class ClusteringConfig(BaseModel):
    enabled: bool = True
    eps: float = 0.5
    min_samples: int = 10

    @field_validator("eps")
    @classmethod
    def validate_eps(cls, v):
        if v <= 0:
            raise ValueError("eps must be > 0")
        return v

    @field_validator("min_samples")
    @classmethod
    def validate_min_samples(cls, v):
        if v < 1:
            raise ValueError("min_samples must be >= 1")
        return v


# ==========================================================
# CLUSTERING INTERPRETATION (EXTENDED DOC)
# ==========================================================
#
# eps:
#   Neighborhood radius (after standardization)
#
#   0.3 → restrictive
#   0.5 → balanced (recommended)
#   0.7 → permissive
#
# min_samples:
#   Minimum density to form cluster
#
#   Low → noise may become cluster
#   High → real structure may disappear
#
# ==========================================================


# ==========================================================
# GEOMETRY CONFIGURATION
# ==========================================================
#
# Defines how the comfort region is converted into a polygon.
#
# ----------------------------------------------------------
# METHODS
# ----------------------------------------------------------
#
# convex:
#   → convex hull
#   → simple
#   → overestimates area
#
# alpha:
#   → alpha shape
#   → non-convex
#   → realistic boundary
#
class GeometryConfig(BaseModel):
    method: str = "alpha"
    alpha: float = 1.2

    @field_validator("method")
    @classmethod
    def validate_method(cls, v):
        if v not in {"alpha", "convex"}:
            raise ValueError("method must be 'alpha' or 'convex'")
        return v

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v):
        if v <= 0:
            raise ValueError("alpha must be > 0")
        return v


# ==========================================================
# GEOMETRY INTERPRETATION (EXTENDED DOC)
# ==========================================================
#
# alpha parameter:
#
#   0.8 → tight, detailed
#   1.2 → balanced (recommended)
#   2.0 → smoother, inflated
#
# Too small:
#   → jagged polygon
#
# Too large:
#   → overly inflated region
#
# ==========================================================


# ==========================================================
# SMOOTHING CONFIGURATION
# ==========================================================
#
# Post-processing of polygon boundaries.
#
# ----------------------------------------------------------
# SCIENTIFIC ROLE
# ----------------------------------------------------------
#
# Improves visualization ONLY.
#
# Does NOT change:
#   - underlying data
#   - statistical interpretation
#
class SmoothingConfig(BaseModel):
    enabled: bool = True
    sigma: float = 2

    @field_validator("sigma")
    @classmethod
    def validate_sigma(cls, v):
        if v <= 0:
            raise ValueError("sigma must be > 0")
        return v


# ==========================================================
# SMOOTHING INTERPRETATION
# ==========================================================
#
# sigma:
#
#   1 → mild
#   2 → balanced
#   3 → strong
#
# Too large:
#   → loss of shape
#
# Too small:
#   → noisy contour remains
#
# ==========================================================


# ==========================================================
# ZONE CONFIGURATION
# ==========================================================
#
# Defines multiple comfort zones based on density percentiles.
#
# ----------------------------------------------------------
# SCIENTIFIC ROLE
# ----------------------------------------------------------
#
# Represents comfort as continuous (NOT binary):
#
#   core → optimal
#   transition → acceptable
#   limit → tolerance boundary
#
class ZonesConfig(BaseModel):
    core_percentile: float = 85
    transition_percentile: float = 60
    limit_percentile: float = 30

    @model_validator(mode="after")
    def validate_order(self):
        if not (
            self.core_percentile >= self.transition_percentile >= self.limit_percentile
        ):
            raise ValueError(
                "Percentiles must satisfy: core >= transition >= limit"
            )
        return self


# ==========================================================
# PIPELINE CONFIGURATION
# ==========================================================
#
# Top-level configuration model.
#
# This replaces the original CONFIG dictionary with:
#
#   → type safety
#   → validation
#   → reproducibility
#
class PipelineConfig(BaseModel):
    dataset_path: Path
    output_fig: Path = Path("fig_comfort_polygon.png")

    density: DensityConfig = DensityConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    geometry: GeometryConfig = GeometryConfig()
    smoothing: SmoothingConfig = SmoothingConfig()
    zones: ZonesConfig = ZonesConfig()

    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_path(cls, v: Path):
        if not v.exists():
            raise ValueError(f"Dataset path does not exist: {v}")
        return v
