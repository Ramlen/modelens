"""Core diagnostic plots for binary classification models."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, PrecisionRecallDisplay


def plot_model_diagnostics(
    model, X, y, X_test, y_test,
    y_pred=None,
    figsize=(16, 10),
    bins=30,
    model_name=None,
    log=False
):
    """
    Comprehensive diagnostic dashboard for binary classification models.

    Parameters
    ----------
    model : object
        Trained model with predict_proba method.
    X : array-like of shape (n_samples, n_features)
        Training features.
    y : array-like of shape (n_samples,)
        Training labels.
    X_test : array-like of shape (n_samples, n_features)
        Test features.
    y_test : array-like of shape (n_samples,)
        Test labels.
    y_pred : array-like of shape (n_samples,), optional
        Predicted labels for test set. Computed if None.
    figsize : tuple, default=(16, 10)
        Figure size.
    bins : int, default=30
        Number of bins for score distribution histograms.
    model_name : str, optional
        Display name. Falls back to type(model).__name__.
    log : bool, default=False
        log-scale for score distribution histograms.

    Returns
    -------
    fig, axs
        Matplotlib figure and axes array.
    """
    if y_pred is None:
        y_pred = model.predict(X_test)

    if model_name is None:
        model_name = type(model).__name__

    y = np.asarray(y)
    y_test = np.asarray(y_test)

    fig, axs = plt.subplots(3, 3, figsize=figsize)

    # Row 0
    plot_loss_curve(model, axs[0, 0])
    plot_confusion_matrix(y_test, y_pred, axs[0, 1])
    plot_roc_curve(model, X_test, y_test, axs[0, 2])

    # Row 1
    plot_score_distribution(model, X, y, axs[1, 0], bins=bins, title="All data", log=log)
    _plot_pr_curve(model, X, y, model_name, pos_label=0, ax=axs[1, 1], title="PR curve, All data (class 0)")
    _plot_pr_curve(model, X, y, model_name, pos_label=1, ax=axs[1, 2], title="PR curve, All data (class 1)")

    # Row 2
    plot_score_distribution(model, X_test, y_test, axs[2, 0], bins=bins, title="Test data", log=log)
    _plot_pr_curve(model, X_test, y_test, model_name, pos_label=0, ax=axs[2, 1], title="PR curve, Test data (class 0)")
    _plot_pr_curve(model, X_test, y_test, model_name, pos_label=1, ax=axs[2, 2], title="PR curve, Test data (class 1)")

    plt.tight_layout()
    return fig, axs


def plot_loss_curve(model, ax):
    """Plot loss curve from model's evals_result_ (CatBoost, XGBoost, LightGBM)."""
    evals = None
    for attr in ("evals_result_", "evals_result", "train_metric_"):
        evals = getattr(model, attr, None)
        if evals:
            break

    if evals:
        try:
            # Try common formats
            for val_key in ("validation", "validation_0"):
                if val_key in evals:
                    for metric in ("Logloss", "logloss", "binary_logloss", "multi_logloss"):
                        if metric in evals[val_key]:
                            ax.plot(evals[val_key][metric])
                            ax.set_xlabel("Iteration")
                            ax.set_ylabel(metric)
                            ax.set_title("Loss")
                            return
            # Fallback: take first metric found
            for val_key in evals:
                if isinstance(evals[val_key], dict):
                    for metric, values in evals[val_key].items():
                        if values and "loss" in metric.lower():
                            ax.plot(values)
                            ax.set_xlabel("Iteration")
                            ax.set_ylabel(metric)
                            ax.set_title("Loss")
                            return
        except Exception:
            pass

    ax.text(0.5, 0.5, "Loss history\nnot available",
            ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_title("Loss")


def plot_score_distribution(model, X, y, ax, bins=30, title="", log=False):
    """Histogram of predicted scores by class."""
    scores = model.predict_proba(X)[:, 1]
    y = np.asarray(y)

    ax.hist(scores[y == 0], alpha=0.7, label="Good", color="dodgerblue", bins=bins)
    ax.hist(scores[y == 1], alpha=0.7, label="Bad", color="darkorange", bins=bins)
    ax.set_xlabel("Score")
    ax.set_ylabel("Counts")
    ax.set_title(title)
    ax.legend(loc="upper center")

    if log:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.0f}'))
        ax.set_ylabel("Counts (log scale)")


def plot_confusion_matrix(y_true, y_pred, ax):
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion matrix")


def plot_roc_curve(model, X_test, y_test, ax):
    """ROC curve with AUC."""
    try:
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.set_title("ROC curve")
    except Exception as e:
        ax.text(0.5, 0.5, f"ROC error:\n{e}", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title("ROC curve")


def plot_pr_curves(model, X, y, X_test, y_test, model_name,
                   ax_all_0, ax_all_1, ax_test_0, ax_test_1):
    """Four Precision-Recall curves (all/test × class 0/1)."""
    _plot_pr_curve(model, X, y, model_name, pos_label=0, ax=ax_all_0, title="PR, All data (class 0)")
    _plot_pr_curve(model, X, y, model_name, pos_label=1, ax=ax_all_1, title="PR, All data (class 1)")
    _plot_pr_curve(model, X_test, y_test, model_name, pos_label=0, ax=ax_test_0, title="PR, Test data (class 0)")
    _plot_pr_curve(model, X_test, y_test, model_name, pos_label=1, ax=ax_test_1, title="PR, Test data (class 1)")


def _plot_pr_curve(model, X, y, model_name, pos_label, ax, title):
    """Single Precision-Recall curve helper."""
    try:
        display = PrecisionRecallDisplay.from_estimator(
            model, X, y, name=model_name,
            plot_chance_level=True, pos_label=pos_label, ax=ax,
        )
        display.ax_.set_title(title)
    except Exception as e:
        ax.text(0.5, 0.5, f"PR error:\n{e}", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title(title)


# Backward-compatible alias
plot_pr_curves.__doc__ = "Plot four Precision-Recall curves (all/test × class 0/1)."