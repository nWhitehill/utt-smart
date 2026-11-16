"""Charts for watching training from training_analysis.ipynb.

Needs the `training` extras (matplotlib, pandas, numpy) and IPython for the live dashboard.
"""
from collections import deque
import math
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Win-rate lines pool this many generations, since one generation's 100
# evaluation games still swing ~10 points either way
ROLLING = 5

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#b4b2a9"  # draws, and the reference lines a loss can't go below
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
# Move probabilities: one hue, from the surface (near 0) to dark blue
HEAT = LinearSegmentedColormap.from_list(
    "move_probability", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

STYLE = {
    "figure.figsize": (11, 4), "figure.dpi": 110, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "lines.linewidth": 2,
    "legend.frameon": False, "legend.labelcolor": INK_2,
}


def use_style():
    plt.rcParams.update(STYLE)


# ---------------------------------------------------------------- the log

def frame(history):
    # The per-generation log as a DataFrame, with the columns the charts use
    log = pd.DataFrame(history)
    if log.empty:
        return log
    log["step"] = range(1, len(log) + 1)  # generation number across all runs in this log
    for outcome in ("wins", "draws", "losses"):
        log[f"eval_{outcome}"] = log[f"eval_x_{outcome}"] + log[f"eval_o_{outcome}"]
    for side in "xo":
        log[f"eval_{side}_games"] = log[[f"eval_{side}_wins", f"eval_{side}_draws", f"eval_{side}_losses"]].sum(axis=1)
    log["eval_games"] = log.eval_x_games + log.eval_o_games
    log["win_rate"] = log.eval_wins / log.eval_games
    log["win_rate_rolling"] = rolling_rate(log.eval_wins, log.eval_games)
    log["win_rate_x_rolling"] = rolling_rate(log.eval_x_wins, log.eval_x_games)
    log["win_rate_o_rolling"] = rolling_rate(log.eval_o_wins, log.eval_o_games)
    games = log["games_per_gen"] if "games_per_gen" in log else log["selfplay_games"]
    log["selfplay_games"] = games
    log["game_length"] = log.selfplay_positions / games
    log["policy_gap"] = log.policy_loss - log.policy_target_entropy
    seconds = [c for c in ("selfplay_seconds", "train_seconds", "eval_seconds", "ladder_seconds") if c in log]
    log["generation_seconds"] = log[seconds].fillna(0).sum(axis=1)
    return log


def rolling_rate(wins, games):
    # Wins over games, pooled across the last ROLLING generations
    return wins.rolling(ROLLING, min_periods=1).sum() / games.rolling(ROLLING, min_periods=1).sum()


def ratings(log):
    # (generation, Elo) for every rated snapshot, starting from the untrained network at 0
    rated = log[log["elo"].notna()] if "elo" in log else log.iloc[:0]
    points = [(0, 0.0)] + list(zip(rated.generation, rated.elo))
    return pd.DataFrame(sorted(set(points)), columns=["generation", "elo"])


def mark_runs(ax, log):
    # A dashed line where each run after the first started, since the settings may have changed
    starts = log.groupby("run", sort=False)["generation"].min()
    for generation in starts.iloc[1:]:
        ax.axvline(generation - 0.5, color=AXIS, linestyle="--", linewidth=1, zorder=0)
    generation_axis(ax)


def generation_axis(ax):
    ax.set_xlabel("Generation")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def end_label(ax, x, y, text):
    # Names a line at its last point, so it isn't identified by colour alone
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2,
                bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"), annotation_clip=False)


def percent_axis(ax):
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")


def waiting(ax, title):
    ax.set_title(title)
    ax.text(0.5, 0.5, "Waits for the first generation", transform=ax.transAxes, ha="center", va="center",
            color=MUTED, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------- charts

def plot_win_rate(ax, log, by_side=False):
    title = "Win rate vs MCT"
    if log.empty:
        return waiting(ax, title)
    ax.scatter(log.generation, log.win_rate, s=14, color=BLUE, alpha=0.45, linewidths=0, label="One generation")
    lines = [("win_rate_rolling", BLUE, "Overall")]
    if by_side:
        lines += [("win_rate_x_rolling", ORANGE, "Playing X"), ("win_rate_o_rolling", AQUA, "Playing O")]
    for column, color, label in lines:
        ax.plot(log.generation, log[column], color=color, label=f"{label}, last {ROLLING} gens")
        end_label(ax, log.generation.iloc[-1], log[column].iloc[-1], f"{label} {log[column].iloc[-1]:.0%}")
    percent_axis(ax)
    mark_runs(ax, log)
    ax.set_title(f"{title}({int(log.mct_iterations.iloc[-1])})")
    ax.legend(loc="upper left", fontsize=8)


def plot_elo(ax, log):
    title = "Elo, from playing earlier snapshots"
    points = ratings(log) if not log.empty else None
    if points is None or len(points) < 2:
        ax.set_title(title)
        ax.text(0.5, 0.5, "Rated at every snapshot", transform=ax.transAxes, ha="center", va="center",
                color=MUTED, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    ax.plot(points.generation, points.elo, color=BLUE, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    end_label(ax, points.generation.iloc[-1], points.elo.iloc[-1], f"{points.elo.iloc[-1]:+.0f}")
    ax.axhline(0, color=NEUTRAL, linewidth=1, linestyle="--", zorder=0)
    mark_runs(ax, log)
    ax.set_ylabel("Elo (untrained network = 0)")
    ax.set_title(title)


def plot_loss(ax, log, kind):
    # kind is "policy" or "value". Each loss is drawn against the lowest value it could reach
    title = f"{kind.capitalize()} loss"
    if log.empty:
        return waiting(ax, title)
    floor, floor_label = (("policy_target_entropy", "Target entropy (floor)") if kind == "policy"
                          else ("value_target_mse", "Predict 0 every time"))
    loss = f"{kind}_loss"
    ax.plot(log.generation, log[loss], color=BLUE, label="Training")
    ax.plot(log.generation, log[f"fresh_{loss}"], color=ORANGE, label="New positions, before training")
    ax.plot(log.generation, log[floor], color=MUTED, linestyle="--", linewidth=1.5, label=floor_label)
    ax.set_ylim(0, log[[loss, floor, f"fresh_{loss}"]].max().max() * 1.15)
    mark_runs(ax, log)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)


def plot_probe_values(ax, probe_history, probes):
    title = "Value the network gives each probe"
    if not probe_history:
        return waiting(ax, title)
    generations = [entry["generation"] for entry in probe_history]
    for probe, color in zip(probes, (BLUE, ORANGE, AQUA)):
        values = [next(a["value"] for a in entry["probes"] if a["name"] == probe.name) for entry in probe_history]
        ax.plot(generations, values, color=color, label=probe.title)
        end_label(ax, generations[-1], values[-1], f"{values[-1]:+.2f}")
    ax.axhline(0, color=NEUTRAL, linewidth=1, linestyle="--", zorder=0)
    ax.set_ylim(-1.05, 1.05)
    generation_axis(ax)
    ax.set_ylabel("Value for the side to move")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=7)


def plot_selfplay(result_ax, length_ax, log):
    bottom = pd.Series(0.0, index=log.index)
    for column, color, label in [("selfplay_x_wins", BLUE, "X wins"), ("selfplay_o_wins", ORANGE, "O wins"),
                                 ("selfplay_draws", NEUTRAL, "Draw")]:
        share = log[column] / log.selfplay_games
        result_ax.bar(log.generation, share, bottom=bottom, width=0.85, color=color, edgecolor=SURFACE,
                      linewidth=1, label=label)
        bottom += share
    percent_axis(result_ax)
    result_ax.set_ylim(0, 1)
    mark_runs(result_ax, log)
    result_ax.set_title("Self-play results")
    result_ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)

    length_ax.plot(log.generation, log.game_length, color=BLUE)
    length_ax.set_ylim(bottom=0)
    mark_runs(length_ax, log)
    length_ax.set_ylabel("Moves per game")
    length_ax.set_title("Self-play game length")


def plot_time(ax, log):
    bottom = pd.Series(0.0, index=log.index)
    for column, color, label in [("selfplay_seconds", BLUE, "Self-play"), ("train_seconds", ORANGE, "Training"),
                                 ("eval_seconds", AQUA, "Evaluation"), ("ladder_seconds", NEUTRAL, "Ladder")]:
        if column not in log:
            continue
        minutes = log[column].fillna(0) / 60
        ax.bar(log.generation, minutes, bottom=bottom, width=0.85, color=color, edgecolor=SURFACE, linewidth=1,
               label=label)
        bottom += minutes
    mark_runs(ax, log)
    ax.set_ylabel("Minutes")
    ax.set_title("Time per generation")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))


# ---------------------------------------------------------------- the board

# The darkest shade is at least this probability, so a policy spread thinly
# over many moves stays pale rather than every cell looking confident
HEAT_FLOOR = 0.25


def draw_board(ax, planes, heat=None, key_move=None, last_move=None, title=None, label_above=0.05):
    # planes is one encoded position (7, 9, 9). heat, if given, is 81 move
    # probabilities shaded over the cells; the larger ones are labelled
    planes = np.asarray(planes, dtype=float)
    x_to_move = planes[6].any()
    x_cells, o_cells = (planes[0], planes[1]) if x_to_move else (planes[1], planes[0])
    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if heat is not None:
        heat = np.asarray(heat, dtype=float).reshape(9, 9)
        top = max(heat.max(), HEAT_FLOOR)
        ax.imshow(heat, cmap=HEAT, vmin=0, vmax=top, extent=(0, 9, 9, 0), interpolation="nearest")
    else:
        top = None
        legal = planes[2]
        for r, c in zip(*np.nonzero(legal)):
            ax.add_patch(Rectangle((c, r), 1, 1, color="#e8f0fb", zorder=0))

    # A subgame that's over gets its owner drawn large and faint across it
    owner_planes = [(planes[3], "X" if x_to_move else "O"), (planes[4], "O" if x_to_move else "X"),
                    (planes[5], "–")]
    for s in range(9):
        r0, c0 = (s // 3) * 3, (s % 3) * 3
        for plane, owner in owner_planes:
            if plane[r0, c0]:
                ax.text(c0 + 1.5, r0 + 1.55, owner, ha="center", va="center", fontsize=40, fontweight="bold",
                        color=ORANGE if owner == "O" else INK if owner == "X" else MUTED, alpha=0.15, zorder=1)

    for i in range(10):
        width = 2 if i % 3 == 0 else 0.5
        ax.plot([0, 9], [i, i], color=INK_2 if i % 3 == 0 else AXIS, linewidth=width, zorder=2)
        ax.plot([i, i], [0, 9], color=INK_2 if i % 3 == 0 else AXIS, linewidth=width, zorder=2)

    for cells, token, color in ((x_cells, "X", INK), (o_cells, "O", ORANGE)):
        for r, c in zip(*np.nonzero(cells)):
            ax.text(c + 0.5, r + 0.54, token, ha="center", va="center", fontsize=10, fontweight="bold",
                    color=color, zorder=3)

    if heat is not None:
        for r, c in zip(*np.nonzero(heat >= label_above)):
            p = heat[r, c]
            ax.text(c + 0.5, r + 0.55, f"{p:.0%}" if p < 0.995 else "99%", ha="center", va="center", fontsize=6.5,
                    color=SURFACE if p > 0.55 * top else INK, zorder=3)
    for move, color in ((last_move, MUTED), (key_move, AQUA)):
        if move is not None:
            ax.add_patch(Rectangle((move % 9 + 0.06, move // 9 + 0.06), 0.88, 0.88, fill=False, edgecolor=color,
                                   linewidth=2, zorder=4))
    if title:
        ax.set_title(title, fontsize=9)


def probe_answer(entry, name):
    return next(a for a in entry["probes"] if a["name"] == name)


def draw_probe(ax, probe, entry, name=True):
    from utt_smart.nn.encoding import encode
    answer = probe_answer(entry, probe.name)
    title = (f"{probe.title}\n" if name else "") + f"gen {entry['generation']} · value {answer['value']:+.2f}"
    if probe.key_move is not None:
        title += ("\n" if not name else " · ") + f"P(cell {probe.key_move}) {answer['policy'][probe.key_move]:.0%}"
    draw_board(ax, encode(probe.game, probe.to_move), heat=answer["policy"], key_move=probe.key_move, title=title)


def probe_evolution(probe, probe_history, columns=6):
    # The network's policy on one probe at evenly spaced generations
    picks = sorted({round(i * (len(probe_history) - 1) / max(1, columns - 1)) for i in range(columns)})
    fig, axes = plt.subplots(1, len(picks), figsize=(2.6 * len(picks), 3.2))
    for ax, i in zip(np.atleast_1d(axes), picks):
        draw_probe(ax, probe, probe_history[i], name=False)
    fig.suptitle(probe.title, x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, -0.02, probe.note, fontsize=8, color=INK_2, ha="left", wrap=True)
    plt.tight_layout()
    return fig


@torch.no_grad()
def network_view(bot, planes, device):
    # The network's own move probabilities and value for one encoded position
    was_training = bot.training
    bot.eval()
    logits, value = bot(torch.as_tensor(planes).float().unsqueeze(0).to(device))
    bot.train(was_training)
    return F.softmax(logits, dim=1)[0].cpu().numpy(), value.item()


def last_moves(record):
    # The move that led to each position, found as the cell that became occupied
    occupied = (record.states[:, 0] + record.states[:, 1]).reshape(len(record.states), 81)
    moves = [None]
    for i in range(1, len(occupied)):
        moves.append(int((occupied[i] - occupied[i - 1]).argmax()))
    return moves


def show_position(bot, record, index, device):
    # One self-play position: what the network thinks now, next to the search's visit counts
    planes = record.states[index]
    prior, value = network_view(bot, planes, device)
    to_move = "X" if planes[6].any() else "O"
    moves = last_moves(record)
    fig, (prior_ax, target_ax) = plt.subplots(1, 2, figsize=(9, 4.6))
    draw_board(prior_ax, planes, heat=prior, last_move=moves[index],
               title=f"Network now (no search)\nvalue {value:+.2f} for {to_move}")
    if record.has_policy[index]:
        draw_board(target_ax, planes, heat=record.policies[index].numpy(), last_move=moves[index],
                   title="Full search's visit counts (the policy target)")
    else:
        draw_board(target_ax, planes, last_move=moves[index],
                   title="Fast search: no policy target,\nvalue target only")
    fig.suptitle(f"Move {index + 1} of {len(record.states)}, {to_move} to move. Game result for {to_move}: "
                 f"{record.values[index].item():+.0f} (the value target)", x=0.01, ha="left", fontsize=10)
    plt.tight_layout()
    return fig


def show_game(record, positions=8):
    # Small boards through one self-play game; the grey square is the move just played
    n = len(record.states)
    picks = sorted({round(i * (n - 1) / max(1, positions - 1)) for i in range(positions)})
    moves = last_moves(record)
    columns = min(4, len(picks))
    rows = math.ceil(len(picks) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.8 * columns, 3.4 * rows))
    for ax in np.atleast_1d(axes).flat[len(picks):]:
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).flat, picks):
        to_move = "X" if record.states[i][6].any() else "O"
        search = "full search" if record.has_policy[i] else "fast search"
        draw_board(ax, record.states[i], last_move=moves[i], title=f"Move {i + 1}: {to_move} to move ({search})")
    winner = {"X": "X won", "O": "O won"}.get(record.winner, "Draw")
    fig.suptitle(f"A self-play game, {n} moves. {winner}", x=0.01, ha="left", fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------- live dashboard

class _Text:
    # Shown as plain text by IPython's display
    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return self.text


class LiveDashboard:
    # Redraws a summary figure after every generation, and shows the trainer's
    # latest messages above it. Takes over trainer.log while it exists
    def __init__(self, trainer, lines=10):
        from IPython.display import display
        self.trainer = trainer
        self.messages = deque(maxlen=lines)
        self.text = display(_Text("Starting"), display_id=True)
        self.figure = display(_Text(""), display_id=True)
        trainer.log = self.log
        self.update(trainer)

    def log(self, message):
        self.messages.append(f"{time.strftime('%H:%M:%S')}  {message}")
        self.text.update(_Text("\n".join(self.messages)))

    def update(self, trainer=None):
        fig = dashboard_figure(self.trainer)
        self.figure.update(fig)
        plt.close(fig)


def dashboard_figure(trainer):
    use_style()
    log = frame(trainer.history)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), gridspec_kw={"height_ratios": [1, 1.15]})
    plot_win_rate(axes[0, 0], log)
    plot_elo(axes[0, 1], log)
    plot_loss(axes[0, 2], log, "policy")
    plot_loss(axes[0, 3], log, "value")
    latest = trainer.probe_history[-1] if trainer.probe_history else None
    for ax, probe in zip(axes[1, :3], trainer.probes):
        if latest is None:
            waiting(ax, probe.title)
        else:
            draw_probe(ax, probe, latest)
    plot_probe_values(axes[1, 3], trainer.probe_history, trainer.probes)
    fig.suptitle(f"Generation {trainer.generation}", x=0.01, ha="left", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig

