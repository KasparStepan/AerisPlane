"""Mission analysis result dataclasses."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SegmentResult:
    """Energy budget for one mission segment."""
    name: str
    duration: float         # seconds
    distance: float         # meters (horizontal)
    energy: float           # Joules consumed
    avg_power: float        # Watts (average)
    avg_speed: float        # m/s
    altitude_start: float   # m
    altitude_end: float     # m


@dataclass
class MissionResult:
    """Complete mission energy budget result."""
    total_energy: float             # Joules consumed
    total_time: float               # seconds
    total_distance: float           # meters
    battery_energy_available: float  # Joules
    energy_margin: float            # fraction remaining (0=empty, 1=full)
    feasible: bool                  # enough battery for the mission
    segments: list[SegmentResult]

    def report(self) -> str:
        lines = []
        lines.append("AerisPlane Mission Analysis")
        lines.append("=" * 75)
        lines.append("")

        header = (
            f"{'Segment':<16} {'Duration':>8} {'Distance':>9} "
            f"{'Energy':>9} {'Avg Power':>10} {'Alt':>10}"
        )
        lines.append(header)
        lines.append("-" * 75)

        for seg in self.segments:
            alt_str = f"{seg.altitude_start:.0f}->{seg.altitude_end:.0f}m"
            lines.append(
                f"{seg.name:<16} {seg.duration:>7.0f}s {seg.distance:>8.0f}m "
                f"{seg.energy / 3600:>8.1f}Wh {seg.avg_power:>9.1f}W {alt_str:>10}"
            )

        lines.append("-" * 75)
        lines.append(
            f"{'TOTAL':<16} {self.total_time:>7.0f}s {self.total_distance:>8.0f}m "
            f"{self.total_energy / 3600:>8.1f}Wh"
        )

        lines.append("")
        lines.append(f"Battery energy:   {self.battery_energy_available / 3600:.1f} Wh")
        lines.append(f"Energy used:      {self.total_energy / 3600:.1f} Wh")
        lines.append(f"Energy margin:    {self.energy_margin * 100:.1f}%")
        lines.append(f"Mission time:     {self.total_time / 60:.1f} min")
        status = "FEASIBLE" if self.feasible else "NOT FEASIBLE"
        lines.append(f"Status:           {status}")

        return "\n".join(lines)

    def plot(self):
        """Mission profile: energy budget bars + altitude profile."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)
        fig, (ax_energy, ax_alt) = plt.subplots(1, 2, figsize=(14, 6))

        # Energy budget bars
        names = [s.name for s in self.segments]
        energies_wh = [s.energy / 3600 for s in self.segments]
        colors = sns.color_palette("husl", n_colors=max(len(self.segments), 1))

        bars = ax_energy.barh(names, energies_wh, color=colors, edgecolor="white")
        for bar, wh in zip(bars, energies_wh):
            ax_energy.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                          f"{wh:.1f} Wh", va="center", fontsize=9)
        ax_energy.set_xlabel("Energy [Wh]")
        ax_energy.set_title("Energy Budget", fontweight="bold")

        # Altitude profile
        t_cumul = 0.0
        for seg in self.segments:
            t_start = t_cumul
            t_end = t_cumul + seg.duration
            ax_alt.plot([t_start / 60, t_end / 60],
                       [seg.altitude_start, seg.altitude_end],
                       "o-", color=PALETTE[0], lw=2, markersize=4)
            ax_alt.text((t_start + t_end) / 2 / 60, (seg.altitude_start + seg.altitude_end) / 2,
                       seg.name, fontsize=8, ha="center", va="bottom")
            t_cumul = t_end

        ax_alt.set_xlabel("Time [min]")
        ax_alt.set_ylabel("Altitude [m]")
        ax_alt.set_title("Altitude Profile", fontweight="bold")
        ax_alt.set_ylim(bottom=-10)

        fig.suptitle(
            f"Mission: {self.total_time/60:.1f} min, "
            f"{self.total_energy/3600:.1f}/{self.battery_energy_available/3600:.1f} Wh "
            f"({'FEASIBLE' if self.feasible else 'INFEASIBLE'})",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(pad=1.0)
        return fig
