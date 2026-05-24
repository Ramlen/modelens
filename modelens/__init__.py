"""modelens — ML model diagnostics lens for binary classification."""

from modelens.diagnostics import plot_model_diagnostics
from modelens.diagnostics import (
    plot_loss_curve,
    plot_score_distribution,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curves,
)

__version__ = "0.1.5"
__all__ = [
    "plot_model_diagnostics",
    "plot_loss_curve",
    "plot_score_distribution",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_pr_curves",
]