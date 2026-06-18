"""
__init__.py — Public API for the semantic_roi_graph package.

Usage:
    from src.models.semantic_roi_graph import SemanticROIGraphFER, SemanticRoiGraphConfig
"""

from .config import SemanticRoiGraphConfig, DEFAULT_SEMANTIC_REGIONS
from .model import (
    SemanticROIGraphFER,
    # Backward-compatible class aliases
    MacroSemanticProgramBank,
    MacroSemanticProgramMatcher,
    MacroMotifBank,
    MacroMotifMatcher,
    SemanticMotifBank,
    SemanticMotifMatcher,
    MicroMotifBank,
    MicroMotifMatcher,
)
from .backbone import SemanticBackbone
from .roi_align import SemanticRoiAlign
from .micro_modules import (
    GATBlock,
    GatedPooling,
    MicroGraphReasoner,
    MicroSemanticMotifBank,
    MicroSemanticMotifMatcher,
)
from .macro_modules import (
    SemanticInteractionBlock,
    CrossRegionCompositionGraph,
    SemanticHypergraphReasoner,
    SemanticCompositionalProgramBank,
    SemanticProgramExecutor,
)
from .classifier import SemanticStateEncoder, SemanticEmotionClassifier
from .utils import safe_softmax

__all__ = [
    # Primary public API
    "SemanticROIGraphFER",
    "SemanticRoiGraphConfig",
    "DEFAULT_SEMANTIC_REGIONS",
    # Sub-modules
    "SemanticBackbone",
    "SemanticRoiAlign",
    "GATBlock",
    "GatedPooling",
    "MicroGraphReasoner",
    "MicroSemanticMotifBank",
    "MicroSemanticMotifMatcher",
    "SemanticInteractionBlock",
    "CrossRegionCompositionGraph",
    "SemanticHypergraphReasoner",
    "SemanticCompositionalProgramBank",
    "SemanticProgramExecutor",
    "SemanticStateEncoder",
    "SemanticEmotionClassifier",
    "safe_softmax",
    # Backward-compatible aliases
    "MacroSemanticProgramBank",
    "MacroSemanticProgramMatcher",
    "MacroMotifBank",
    "MacroMotifMatcher",
    "SemanticMotifBank",
    "SemanticMotifMatcher",
    "MicroMotifBank",
    "MicroMotifMatcher",
]
