import numpy as np
from matplotlib import pyplot as plt


def plot_force_alignment_prob(melspec,
                              ph_seq,
                              ph_intervals,
                              frame_confidence,
                              edge_prob,
                              ph_frame_prob,
                              ph_frame_id_gt,
                              ph_time_gt=None,
                              v_min=-8, v_max=2
                              ):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})

    ph_seq = [i.split("/")[-1] for i in ph_seq]
    T = melspec.shape[-1]
    x = np.arange(T)

    ax1.pcolormesh(x, np.arange(melspec.shape[0]), melspec, shading='auto', vmin=v_min, vmax=v_max)

    y_max = melspec.shape[0]
    ax1.set_ylim(0, y_max)

    draw_upper_blue = ph_time_gt is not None
    red_upper, red_full = [], []

    all_points = []
    for interval in ph_intervals:
        all_points.extend([interval[0], interval[1]])
    all_points = np.array(all_points)

    valid_points = all_points[(all_points >= 0) & (all_points < T)]
    if draw_upper_blue:
        red_upper = valid_points
    else:
        red_full = valid_points

    if len(red_upper) > 0:
        ax1.vlines(red_upper, ymin=0.5 * y_max, ymax=y_max, colors='r', linewidth=1, zorder=2)
    if len(red_full) > 0:
        ax1.vlines(red_full, ymin=0, ymax=y_max, colors='r', linewidth=1, zorder=2)

    for i, interval in enumerate(ph_intervals):
        if ph_seq[i] == "SP":
            continue
        y_offset = len(ph_seq[i]) * T / 275
        x_center = (interval[0] + interval[1]) / 2
        ax1.text(
            x_center - y_offset,
            y_max + 1 if i % 2 else y_max - 6,
            ph_seq[i],
            fontsize=11,
            color="black" if i % 2 else "white",
            verticalalignment='bottom' if i % 2 else 'top',
            zorder=1
        )

    if draw_upper_blue:
        valid_times = [t for t in ph_time_gt if 0 <= t < T]
        ax1.vlines(valid_times, ymin=0, ymax=0.5 * y_max, colors='b', linewidth=1, zorder=2)

    y_scale = y_max
    ax1.plot(x, frame_confidence * y_scale, 'k-', lw=1, alpha=0.6, zorder=3)
    ax1.fill_between(x, frame_confidence * y_scale, color='k', alpha=0.3, zorder=3)

    legend_elements = [
        plt.Line2D([0], [0], color='r', lw=2, label='pred'),
        plt.Line2D([0], [0], color='b', lw=2, label='gt')
    ]
    ax1.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=10,
        framealpha=0.9,
        ncol=1
    )

    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Mel Bin')

    ax2.imshow(
        ph_frame_prob,
        origin="lower",
        aspect="auto",
        interpolation="nearest"
    )

    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Probability')

    ax2.plot(x, ph_frame_id_gt, color="red", linewidth=1.5)
    ax2.plot(x, edge_prob * ph_frame_prob.shape[0], color="black", linewidth=1)
    ax2.fill_between(x, edge_prob * ph_frame_prob.shape[0], color="black", alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    return fig


def find_continuous_regions(row):
    if np.sum(row) == 0:
        return []

    one_indices = np.where(row == 1)[0]

    if len(one_indices) == 0:
        return []

    diff = np.diff(one_indices)
    break_points = np.where(diff > 1)[0] + 1

    regions = np.split(one_indices, break_points)
    return [(region[0], region[-1]) for region in regions if len(region) > 0]


def plot_non_lexical_phonemes(mel_spec,  # [C,T]
                              cvnt_prob,
                              label=None,
                              non_lexical_target=None,  # 形状为 [N, T]
                              v_min=-8, v_max=2,
                              bar_alpha=0.7, pcolor_alpha=0.4, frame_duration=None):
    label = label or [f'Tensor {i}' for i in range(len(cvnt_prob))]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})

    if mel_spec.ndim == 3:
        mel_spec = mel_spec.squeeze(0)

    C, T = mel_spec.shape
    X, Y = np.meshgrid(np.arange(T + 1), np.arange(C + 1))
    ax1.pcolormesh(X, Y, mel_spec, shading='flat', vmin=v_min, vmax=v_max)

    if non_lexical_target is not None:
        N, T_target = non_lexical_target.shape

        if T_target != T:
            print(f"Warning: non_lexical_target 的时间维度 ({T_target}) 与 mel_spec 的时间维度 ({T}) 不匹配")
            T_min = min(T, T_target)
        else:
            T_min = T

        all_boundaries = []

        if N > 1:
            target_rows = non_lexical_target[1:, :T_min]
            for i, target_row in enumerate(target_rows):
                regions = find_continuous_regions(target_row)

                for region_start, region_end in regions:
                    all_boundaries.extend([region_start, region_end + 1])

            all_boundaries = np.unique(all_boundaries)
            valid_boundaries = all_boundaries[(all_boundaries >= 0) & (all_boundaries < T)]

            if len(valid_boundaries) > 0:
                ax1.vlines(valid_boundaries, ymin=0, ymax=C, colors='b', linewidth=1, zorder=2)

            for i, target_row in enumerate(target_rows):
                regions = find_continuous_regions(target_row)

                for region_start, region_end in regions:
                    region_center = (region_start + region_end) / 2
                    ph_name = label[i + 1] if (i + 1) < len(label) else f"Ph{i + 1}"
                    y_offset = len(ph_name) * T / 275
                    x_center = region_center - y_offset

                    ax1.text(
                        x_center,
                        C + 1,
                        ph_name,
                        fontsize=11,
                        color="black",
                        verticalalignment='bottom',
                        zorder=1
                    )

    if frame_duration is not None:
        time_axis = np.arange(T) * frame_duration
        time_edges = np.linspace(0, T * frame_duration, T + 1)
    else:
        time_axis = np.arange(T)
        time_edges = np.arange(T + 1)

    scaled_prob = cvnt_prob * C
    cumulative = np.zeros(T)

    for i in range(len(scaled_prob)):
        if i == 0:
            cumulative += scaled_prob[0]
        else:
            ax2.fill_between(
                time_axis,
                cumulative,
                cumulative + scaled_prob[i],
                alpha=bar_alpha,
                label=label[i]
            )
            cumulative += scaled_prob[i]

    X_edges, Y_edges = np.meshgrid(time_edges, np.arange(C + 1))
    ax2.pcolormesh(
        X_edges,
        Y_edges,
        mel_spec,
        vmin=v_min,
        vmax=v_max,
        alpha=pcolor_alpha,
        shading='flat'
    )

    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Mel Bin')
    ax2.set_xlabel('Time (s)' if frame_duration else 'Frame Index')
    ax2.set_ylabel('Probability')

    if any(lbl is not None for lbl in label):
        ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    return fig
