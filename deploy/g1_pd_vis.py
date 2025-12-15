from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Any

import fire
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import yaml
from matplotlib.colors import Normalize


def main(
    dof_name: str = "elbow",
    sim: bool = True,
    x_axis: str = "dof_pos_lag",
    y_axis: str = "dof_vel_SD",
    common_parameter: str = "pd",
    plot_parameter: str = "feed_forward_ratio",
    plot_variables: dict[str, float] | None = None,
) -> None:
    if plot_variables is None:
        plot_variables = {"low_pass_alpha": 1.0}

    file_prefix = f"{dof_name}_{'sim' if sim else 'real'}"
    log_dir = Path(__file__).parent / "logs" / "pd_test"
    yaml_files = list((log_dir / "yaml").glob(f"{file_prefix}*.yaml"))

    # Load and filter data
    data = []
    for yaml_file in yaml_files:
        with open(yaml_file) as f:
            log = yaml.load(f, Loader=yaml.FullLoader)
            # Filter by plot_variables
            match = True
            for plot_variable, value in plot_variables.items():
                if log.get(plot_variable) != value:
                    print(f"Skipping {yaml_file} because {plot_variable} is not {value}")
                    match = False
                    break
            if match:
                data.append(log)

    if not data:
        print("No data found matching the criteria.")
        return

    # Identify unmentioned parameters (not in plot_variables, common_parameter, or plot_parameter)
    mentioned_params = set(plot_variables.keys()) | {common_parameter, plot_parameter}

    # If common_parameter is "pd", also exclude kp and kd since they form pd
    if common_parameter == "pd":
        mentioned_params |= {"kp", "kd"}

    # Get all parameter names from the data
    all_params = set()
    for log in data:
        all_params.update(log.keys())

    # Remove non-parameter keys (metrics like dof_pos_lag, dof_vel_SD, etc.)
    parameter_keys = {"period", "kp", "kd", "feed_forward_ratio", "low_pass_alpha"}
    all_params = {p for p in all_params if p in parameter_keys}

    unmentioned_params = sorted(all_params - mentioned_params)

    # Group data by unmentioned parameters
    grouped_data = defaultdict(list)
    for log in data:
        # Create key from unmentioned parameters
        if unmentioned_params:
            key_parts = [log.get(param) for param in unmentioned_params]
            key = tuple(key_parts) if len(key_parts) > 1 else key_parts[0]
        else:
            key = None
        grouped_data[key].append(log)

    # Create a figure for each group
    for group_key, group_data in grouped_data.items():
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get all plot_parameter values to set up colormap
        all_plot_param_values = [log.get(plot_parameter, 0) for log in group_data]
        plot_param_min = min(all_plot_param_values)
        plot_param_max = max(all_plot_param_values)

        # Create colormap for plot_parameter
        if plot_param_min == plot_param_max:
            # Single value, use a default colormap
            cmap = cm.get_cmap("viridis")
            norm = Normalize(vmin=plot_param_min, vmax=plot_param_max + 0.1)
        else:
            cmap = cm.get_cmap("viridis")
            norm = Normalize(vmin=plot_param_min, vmax=plot_param_max)

        # Group by common_parameter
        common_groups = defaultdict(list)
        for log in group_data:
            if common_parameter == "pd":
                common_key = (log.get("kp"), log.get("kd"))
            else:
                common_key = log.get(common_parameter)
            common_groups[common_key].append(log)

        # Sort common groups for consistent ordering
        sorted_common_groups = sorted(common_groups.items())

        # Color cycle for line colors (lighter/gray for lines)
        color_list = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        line_colors = cycle(color_list)

        scatter_mappable = None
        for common_key, common_data in sorted_common_groups:
            # Sort by plot_parameter
            common_data_sorted = sorted(common_data, key=lambda x: x.get(plot_parameter, 0))

            # Extract x and y values
            x_values = [log.get(x_axis) for log in common_data_sorted]
            y_values = [log.get(y_axis) for log in common_data_sorted]
            plot_param_values = [log.get(plot_parameter, 0) for log in common_data_sorted]

            # Get line color (lighter for connecting lines)
            line_color = next(line_colors)

            # Plot line connecting points (lighter color)
            ax.plot(
                x_values,
                y_values,
                "-",
                color=line_color,
                linewidth=1.5,
                alpha=0.5,
                label=_format_label(common_parameter, common_key),
            )

            # Plot points colored by plot_parameter (using cross markers)
            scatter = ax.scatter(
                x_values,
                y_values,
                c=plot_param_values,
                cmap=cmap,
                norm=norm,
                s=100,
                marker="x",
                linewidths=2,
                zorder=5,
            )
            scatter_mappable = scatter

        # Set labels and title
        ax.set_xlabel(x_axis, fontsize=12)
        ax.set_ylabel(y_axis, fontsize=12)

        # Create title from group_key
        if group_key is not None and unmentioned_params:
            title_parts = []
            if isinstance(group_key, tuple):
                for i, param in enumerate(unmentioned_params):
                    val = group_key[i]
                    title_parts.append(f"{param}={val}")
            else:
                title_parts.append(f"{unmentioned_params[0]}={group_key}")
            title = ", ".join(title_parts)
        else:
            title = "All data"

        ax.set_title(title, fontsize=14)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add colorbar for plot_parameter
        if scatter_mappable is not None:
            cbar = plt.colorbar(scatter_mappable, ax=ax)
            cbar.set_label(plot_parameter, fontsize=12)

        # Save figure
        output_dir = log_dir / "plots"
        output_dir.mkdir(exist_ok=True)

        # Create filename from group_key
        if group_key is not None and unmentioned_params:
            filename_parts = [file_prefix]
            if isinstance(group_key, tuple):
                for i, param in enumerate(unmentioned_params):
                    val = group_key[i]
                    filename_parts.append(f"{param}_{val}")
            else:
                filename_parts.append(f"{unmentioned_params[0]}_{group_key}")
            filename = "_".join(filename_parts) + ".png"
        else:
            filename = f"{file_prefix}_all.png"

        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
        plt.close()


def _format_label(common_parameter: str, common_key: Any) -> str:
    """Format label for legend."""
    if common_parameter == "pd":
        kp, kd = common_key
        return f"PD: kp={kp}, kd={kd}"
    else:
        return f"{common_parameter}={common_key}"


if __name__ == "__main__":
    fire.Fire(main)
