"""Visualization utilities for adversarial dust experiments."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

def plot_degradation_curves(results: dict, output_path: str):
    """Plot success rate vs budget level: adversarial, random, clean.

    Args:
        results: Output from BudgetSweep.run().
        output_path: Path to save the plot.
    """
    budget_results = results["budget_results"]
    clean_sr = results["clean_sr"]

    budgets = sorted(float(b) for b in budget_results.keys())
    adv_srs = [budget_results[str(b)]["adversarial_sr"] for b in budgets]
    rand_means = [budget_results[str(b)]["random_mean_sr"] for b in budgets]
    rand_stds = [budget_results[str(b)]["random_std_sr"] for b in budgets]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Clean baseline
    ax.axhline(clean_sr, color="green", linestyle="--", linewidth=2, label="Clean (no dust)")

    # Random baselines with error band
    ax.plot(budgets, rand_means, "b-o", linewidth=2, label="Random dust (mean)")
    ax.fill_between(
        budgets,
        [m - s for m, s in zip(rand_means, rand_stds)],
        [m + s for m, s in zip(rand_means, rand_stds)],
        alpha=0.2,
        color="blue",
    )

    # Adversarial
    ax.plot(budgets, adv_srs, "r-s", linewidth=2, label="Adversarial dust")

    ax.set_xlabel("Dust Budget (pixel coverage fraction)", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Policy Degradation Under Camera Dust", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_dust_pattern(
    dust_model,
    params: np.ndarray,
    sample_image: np.ndarray,
    output_path: str,
    timestep: int = 0,
):
    """3-panel visualization: clean image | alpha mask heatmap | dirty image.

    Args:
        dust_model: Any dust model with get_alpha_mask() and apply().
        params: Flat array of dust parameters.
        sample_image: Clean input image (uint8 HxWx3).
        output_path: Path to save the plot.
        timestep: Timestep for dynamic models (ignored by grid model).
    """
    alpha_mask = dust_model.get_alpha_mask(params, timestep=timestep)
    dirty_image = dust_model.apply(sample_image, params, timestep=timestep)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(sample_image)
    axes[0].set_title("Clean Image")
    axes[0].axis("off")

    max_opacity = getattr(dust_model.config, "max_cell_opacity", getattr(dust_model.config, "max_opacity", 0.6))
    im = axes[1].imshow(alpha_mask, cmap="hot", vmin=0, vmax=max_opacity)
    axes[1].set_title("Dust Alpha Mask")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(dirty_image)
    axes[2].set_title("Dirty Image")
    axes[2].axis("off")

    coverage = dust_model.compute_coverage(alpha_mask)
    fig.suptitle(f"Coverage: {coverage:.1%}", fontsize=13)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_budget_comparison(
    results: dict,
    image_shape: tuple,
    dust_config,
    sample_image: np.ndarray,
    output_path: str,
    model_factory=None,
):
    """Grid of adversarial dust patterns across all budget levels.

    Args:
        results: Output from BudgetSweep.run().
        image_shape: (H, W, C) of the images.
        dust_config: DustGridConfig or BlobConfig instance.
        sample_image: Clean image to apply patterns to.
        output_path: Path to save the plot.
        model_factory: Optional callable(image_shape, budget) -> dust_model.
            If None, creates an AdversarialDustModel (grid) for backward compat.
    """
    budget_results = results["budget_results"]
    budgets = sorted(float(b) for b in budget_results.keys())
    n_budgets = len(budgets)

    fig, axes = plt.subplots(2, n_budgets, figsize=(5 * n_budgets, 10))
    if n_budgets == 1:
        axes = axes[:, np.newaxis]

    if model_factory is None:
        from adversarial_dust.dust_model import AdversarialDustModel
        model_factory = lambda shape, budget: AdversarialDustModel(dust_config, shape, budget)

    for col, budget in enumerate(budgets):
        br = budget_results[str(budget)]
        params = np.array(br["adversarial_params"])

        dust_model = model_factory(image_shape, budget)
        alpha_mask = dust_model.get_alpha_mask(params)
        dirty_image = dust_model.apply(sample_image, params)

        max_opacity = getattr(dust_config, "max_cell_opacity", getattr(dust_config, "max_opacity", 0.6))
        axes[0, col].imshow(alpha_mask, cmap="hot", vmin=0, vmax=max_opacity)
        axes[0, col].set_title(f"Budget={budget:.0%}\nSR={br['adversarial_sr']:.2f}")
        axes[0, col].axis("off")

        axes[1, col].imshow(dirty_image)
        axes[1, col].set_title("Dirty Image")
        axes[1, col].axis("off")

    fig.suptitle("Adversarial Dust Patterns Across Budget Levels", fontsize=14)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_adversarial_training_curves(history: dict, output_path: str):
    """Plot adversarial training curves: success rates over rounds.

    Creates a multi-panel figure showing:
    - Top: adversarial and clean success rates per round (one line per budget)
    - Bottom: attacker final success rate per round

    Args:
        history: Dict from AdversarialTrainer.get_history_dict() with keys:
            rounds, budgets, adversarial_srs, clean_srs, attacker_final_srs.
        output_path: Path to save the plot.
    """
    rounds = np.array(history["rounds"])
    budgets = np.array(history["budgets"])
    adv_srs = np.array(history["adversarial_srs"])
    clean_srs = np.array(history["clean_srs"])
    attacker_srs = np.array(history["attacker_final_srs"])

    unique_budgets = sorted(set(budgets))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top panel: eval success rates
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_budgets)))

    for i, budget in enumerate(unique_budgets):
        mask = budgets == budget
        r = rounds[mask]
        ax.plot(
            r, adv_srs[mask], "-s", color=colors[i], linewidth=2,
            label=f"Adversarial (budget={budget:.0%})",
        )
        ax.plot(
            r, clean_srs[mask], "--o", color=colors[i], linewidth=1.5,
            alpha=0.5, label=f"Clean (budget={budget:.0%})",
        )

    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Zero-Sum Adversarial Training: Eval Success Rates", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Bottom panel: attacker training success rate
    ax = axes[1]
    for i, budget in enumerate(unique_budgets):
        mask = budgets == budget
        r = rounds[mask]
        ax.plot(
            r, attacker_srs[mask], "-^", color=colors[i], linewidth=2,
            label=f"Budget={budget:.0%}",
        )

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Attacker Success Rate\n(lower = stronger attack)", fontsize=11)
    ax.set_title("Generator Training Progress", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dust-robustness comparison (baseline vs fine-tuned)
# ---------------------------------------------------------------------------


def plot_operating_envelope(results: dict, output_path: str):
    """Plot the safe operating envelope across all occlusion types.

    Generates a multi-curve plot showing worst-case success rate vs occlusion
    budget for each occlusion type, with safe/marginal/unsafe zones shaded.

    Args:
        results: Output from SafeOperatingEnvelopePredictor.predict().
        output_path: Path to save the plot.
    """
    occlusion_results = results["occlusion_results"]
    clean_sr = results["clean_sr"]
    safe_thresh = results["safe_threshold"]
    marginal_thresh = results["marginal_threshold"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Shade zones
    ax.axhspan(safe_thresh, 1.05, alpha=0.08, color="green", label="_nolegend_")
    ax.axhspan(marginal_thresh, safe_thresh, alpha=0.08, color="gold", label="_nolegend_")
    ax.axhspan(-0.05, marginal_thresh, alpha=0.08, color="red", label="_nolegend_")

    # Zone labels on right margin
    ax.text(
        1.02, (safe_thresh + 1.0) / 2, "SAFE",
        transform=ax.get_yaxis_transform(), fontsize=9, fontweight="bold",
        color="green", va="center",
    )
    ax.text(
        1.02, (marginal_thresh + safe_thresh) / 2, "MARGINAL",
        transform=ax.get_yaxis_transform(), fontsize=9, fontweight="bold",
        color="goldenrod", va="center",
    )
    ax.text(
        1.02, marginal_thresh / 2, "UNSAFE",
        transform=ax.get_yaxis_transform(), fontsize=9, fontweight="bold",
        color="red", va="center",
    )

    # Threshold lines
    ax.axhline(safe_thresh, color="green", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(marginal_thresh, color="red", linestyle=":", alpha=0.5, linewidth=1)

    # Clean baseline
    ax.axhline(clean_sr, color="gray", linestyle="--", linewidth=2, label="Clean (no occlusion)")

    # Style map per occlusion type
    styles = {
        "fingerprint": {"color": "#8B4513", "marker": "o", "linestyle": "-"},
        "glare": {"color": "#FFD700", "marker": "^", "linestyle": "-"},
        "grid": {"color": "#808080", "marker": "s", "linestyle": "--"},
        "blob": {"color": "#4169E1", "marker": "D", "linestyle": "-."},
    }
    default_style = {"color": "tab:purple", "marker": "x", "linestyle": "-"}

    for occ_type, type_results in occlusion_results.items():
        budgets = sorted(float(b) for b in type_results.keys())
        adv_srs = [type_results[str(b)]["adversarial_sr"] for b in budgets]
        rand_means = [type_results[str(b)]["random_mean_sr"] for b in budgets]
        rand_stds = [type_results[str(b)]["random_std_sr"] for b in budgets]

        style = styles.get(occ_type, default_style)
        label_name = occ_type.replace("_", " ").title()

        # Adversarial (worst-case)
        ax.plot(
            budgets, adv_srs,
            linewidth=2.5, markersize=7,
            label=f"{label_name} (adversarial)",
            **style,
        )

        # Random baseline (lighter)
        ax.plot(
            budgets, rand_means,
            linewidth=1, alpha=0.5,
            color=style["color"], linestyle=":",
            label=f"{label_name} (random)",
        )
        ax.fill_between(
            budgets,
            [m - s for m, s in zip(rand_means, rand_stds)],
            [m + s for m, s in zip(rand_means, rand_stds)],
            alpha=0.1, color=style["color"],
        )

    ax.set_xlabel("Occlusion Budget (pixel coverage fraction)", fontsize=12)
    ax.set_ylabel("Worst-Case Success Rate", fontsize=12)
    ax.set_title(
        f"Safe Operating Envelope: {results['task']}\n"
        f"Policy: {results['policy']}",
        fontsize=14,
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    # Add envelope summary annotations
    envelope = results.get("envelope_summary", {})
    y_offset = 0.95
    for occ_type, summary in envelope.items():
        max_safe = summary["max_safe_budget"]
        label_name = occ_type.replace("_", " ").title()
        if max_safe > 0:
            ax.annotate(
                f"{label_name}: safe <= {max_safe:.0%}",
                xy=(max_safe, safe_thresh), fontsize=8,
                xytext=(max_safe + 0.02, y_offset),
                textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="->", color="gray"),
            )
            y_offset -= 0.06

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_occlusion_gallery(
    results: dict,
    image_shape: tuple,
    config,
    sample_image: np.ndarray,
    output_path: str,
):
    """Grid of worst-case occlusion patterns: rows=types, cols=budgets.

    Args:
        results: Output from SafeOperatingEnvelopePredictor.predict().
        image_shape: (H, W, C).
        config: EnvelopeExperimentConfig.
        sample_image: Clean image to apply patterns to.
        output_path: Path to save the plot.
    """
    from adversarial_dust.envelope_predictor import make_occlusion_model

    occlusion_results = results["occlusion_results"]
    occ_types = list(occlusion_results.keys())
    budgets = sorted(
        set(float(b) for tr in occlusion_results.values() for b in tr.keys())
    )

    n_rows = len(occ_types)
    n_cols = len(budgets)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, occ_type in enumerate(occ_types):
        type_results = occlusion_results[occ_type]
        for col, budget in enumerate(budgets):
            ax = axes[row, col]
            budget_key = str(budget)
            if budget_key not in type_results:
                ax.axis("off")
                continue

            br = type_results[budget_key]
            params = np.array(br["adversarial_params"])
            model = make_occlusion_model(occ_type, config, image_shape, budget)
            dirty = model.apply(sample_image, params)

            ax.imshow(dirty)
            zone = br["zone"]
            zone_color = {"safe": "green", "marginal": "goldenrod", "unsafe": "red"}[zone]
            ax.set_title(
                f"SR={br['adversarial_sr']:.2f} [{zone}]",
                fontsize=9, color=zone_color, fontweight="bold",
            )
            ax.axis("off")

            if col == 0:
                ax.set_ylabel(occ_type.replace("_", " ").title(), fontsize=11)
            if row == 0:
                ax.set_title(
                    f"Budget={budget:.0%}\nSR={br['adversarial_sr']:.2f} [{zone}]",
                    fontsize=9, color=zone_color, fontweight="bold",
                )

    fig.suptitle("Adversarial Occlusion Gallery: Worst-Case Patterns", fontsize=14)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_comparison(
    results: dict,
    output_path: str,
    severity_order: Optional[List[str]] = None,
):
    """Plot severity vs success_rate for baseline and fine-tuned models.

    Args:
        results: Nested dict from ``DustRobustnessEvaluator.evaluate_all()``.
            Structure: results[model_name][severity_name]["success_rate"].
        output_path: Path to save the plot.
        severity_order: Explicit ordering of severity levels on x-axis.
            Defaults to ``["clean", "light", "moderate", "heavy"]``.
    """
    if severity_order is None:
        severity_order = ["clean", "light", "moderate", "heavy"]

    model_names = list(results.keys())

    # Style mapping for up to 4 models
    styles = [
        {"color": "tab:blue", "linestyle": "--", "marker": "o"},
        {"color": "tab:orange", "linestyle": "-", "marker": "s"},
        {"color": "tab:green", "linestyle": "-.", "marker": "^"},
        {"color": "tab:red", "linestyle": ":", "marker": "D"},
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = np.arange(len(severity_order))

    for i, model_name in enumerate(model_names):
        style = styles[i % len(styles)]
        model_results = results[model_name]

        srs = []
        errs = []
        for sev in severity_order:
            if sev in model_results:
                sr = model_results[sev]["success_rate"]
                successes = model_results[sev].get("successes", [])
                srs.append(sr)
                # Standard error of the mean
                if len(successes) > 1:
                    se = np.std(successes, ddof=1) / np.sqrt(len(successes))
                else:
                    se = 0.0
                errs.append(se)
            else:
                srs.append(np.nan)
                errs.append(0.0)

        ax.errorbar(
            x_positions,
            srs,
            yerr=errs,
            label=model_name,
            linewidth=2,
            capsize=4,
            **style,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(severity_order, fontsize=11)
    ax.set_xlabel("Dust Severity", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title(
        "Policy Robustness Under Camera Dust\n(Baseline vs Fine-Tuned)",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
