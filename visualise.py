"""
VRPTW Island GA — Visualisation Module
========================================
Produces four visualisations from a solved VRPTWSolver:

    1. Route Map          — 2D plot of all vehicle routes
    2. Convergence Chart  — fitness per generation per island
    3. Solution Dashboard — summary metrics and route breakdown
    4. Time Window Chart  — arrival times vs time windows per customer

Usage (standalone):
    python visualise.py

Usage (from code):
    from visualise import Visualiser
    vis = Visualiser(solver)
    vis.show_all()
    vis.save_all("output/")
"""

import os
import sys
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines   import Line2D

# use Agg backend for saving charts — works on all platforms
# without needing a display. Change to 'TkAgg' or 'Qt5Agg'
# if you want interactive popup windows on your machine.
matplotlib.use('TkAgg')


# ── colour palette — one colour per vehicle ───────────────────────────────────
VAN_COLOURS = [
    '#2E75B6',  # blue
    '#1D9E75',  # green
    '#E07B39',  # orange
    '#9B59B6',  # purple
    '#E74C3C',  # red
    '#1ABC9C',  # teal
    '#F39C12',  # yellow
    '#2C3E50',  # dark blue
    '#8E44AD',  # dark purple
    '#27AE60',  # dark green
]


class Visualiser:
    """
    Builds all visualisations from a solved VRPTWSolver instance.
    """

    def __init__(self, solver):
        """
        Args:
            solver: A VRPTWSolver instance after solve() has been called.
        """
        self.solver    = solver
        self.schedule  = solver.get_schedule()
        self.customers = solver.get_customers()
        self.depot     = solver.get_depot()
        self.solution  = solver.get_solution()
        self.manager   = solver.get_island_manager()
        self.params    = solver.params

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC INTERFACE
    # ══════════════════════════════════════════════════════════════════════════

    def show_all(self) -> None:
        """
        Displays all four charts in separate windows.
        Call after solve() has completed.
        """
        self.plot_route_map()
        self.plot_convergence()
        self.plot_time_windows()
        self.plot_dashboard()
        plt.show()

    def save_all(self, output_dir: str = "output") -> None:
        """
        Saves all four charts as PNG files without displaying them.

        Args:
            output_dir: Directory to save the files into.
        """
        os.makedirs(output_dir, exist_ok=True)

        self.plot_route_map(show=False)
        plt.savefig(os.path.join(output_dir, "route_map.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir}/route_map.png")

        self.plot_convergence(show=False)
        plt.savefig(os.path.join(output_dir, "convergence.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir}/convergence.png")

        self.plot_time_windows(show=False)
        plt.savefig(os.path.join(output_dir, "time_windows.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir}/time_windows.png")

        self.plot_dashboard(show=False)
        plt.savefig(os.path.join(output_dir, "dashboard.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir}/dashboard.png")

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 1 — ROUTE MAP
    # ══════════════════════════════════════════════════════════════════════════

    def plot_route_map(self, show: bool = True) -> None:
        """
        Draws the 2D delivery route map.

        Shows:
            - Depot as a large black star
            - Each customer as a numbered circle
            - Each vehicle's route as a coloured line with arrows
            - Legend identifying each van
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#F0F4F8')

        ax.set_title("VRPTW — Optimised Delivery Routes",
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("X Coordinate", fontsize=11)
        ax.set_ylabel("Y Coordinate", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        depot_x, depot_y = self.depot.get_location()

        # ── draw each vehicle route ────────────────────────────────────────
        legend_handles = []

        for v_idx, vehicle in enumerate(self.schedule):
            colour = VAN_COLOURS[v_idx % len(VAN_COLOURS)]
            route  = vehicle['route']

            # build coordinate path: depot → customers → depot
            xs = [depot_x]
            ys = [depot_y]
            for cid in route:
                cx, cy = self.customers[cid].get_location()
                xs.append(cx)
                ys.append(cy)
            xs.append(depot_x)
            ys.append(depot_y)

            # draw route line
            ax.plot(xs, ys, '-', color=colour, linewidth=1.8,
                    alpha=0.7, zorder=2)

            # draw arrows along the route to show direction
            for i in range(len(xs) - 1):
                dx = xs[i+1] - xs[i]
                dy = ys[i+1] - ys[i]
                mid_x = xs[i] + dx * 0.5
                mid_y = ys[i] + dy * 0.5
                ax.annotate('',
                    xy=(mid_x + dx * 0.01, mid_y + dy * 0.01),
                    xytext=(mid_x - dx * 0.01, mid_y - dy * 0.01),
                    arrowprops=dict(arrowstyle='->', color=colour,
                                   lw=1.5),
                    zorder=3
                )

            # legend entry
            legend_handles.append(
                Line2D([0], [0], color=colour, linewidth=2,
                       label=f"Van {v_idx + 1} "
                             f"({vehicle['customers_served']} stops, "
                             f"dist={vehicle['total_distance']:.1f})")
            )

        # ── draw customers ─────────────────────────────────────────────────
        for cid, customer in self.customers.items():
            cx, cy = customer.get_location()

            # find which van visits this customer for colouring
            van_colour = '#555555'
            for v_idx, vehicle in enumerate(self.schedule):
                if cid in vehicle['route']:
                    van_colour = VAN_COLOURS[v_idx % len(VAN_COLOURS)]
                    break

            ax.scatter(cx, cy, s=120, color=van_colour,
                       edgecolors='white', linewidths=1.5,
                       zorder=4)
            ax.annotate(str(cid),
                        xy=(cx, cy),
                        xytext=(4, 4),
                        textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        color='#222222', zorder=5)

        # ── draw depot ─────────────────────────────────────────────────────
        ax.scatter(depot_x, depot_y, s=300, color='#1A1A2E',
                   marker='*', zorder=6, label='Depot')
        ax.annotate('DEPOT',
                    xy=(depot_x, depot_y),
                    xytext=(6, 6),
                    textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    color='#1A1A2E', zorder=7)

        # add depot to legend
        legend_handles.insert(0,
            Line2D([0], [0], marker='*', color='w',
                   markerfacecolor='#1A1A2E', markersize=12,
                   label='Depot'))

        ax.legend(handles=legend_handles, loc='upper right',
                  fontsize=9, framealpha=0.9)

        # metrics annotation
        total_dist = sum(v['total_distance'] for v in self.schedule)
        total_viol = sum(v['total_violation'] for v in self.schedule)
        info = (f"Vans: {len(self.schedule)}  |  "
                f"Total dist: {total_dist:.1f}  |  "
                f"Violations: {total_viol:.2f}  |  "
                f"Fitness: {self.solution.get_fitness():.2f}")
        ax.text(0.01, 0.01, info, transform=ax.transAxes,
                fontsize=9, color='#333333',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', alpha=0.8))

        plt.tight_layout()
        if show:
            plt.show()

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 2 — CONVERGENCE CHART
    # ══════════════════════════════════════════════════════════════════════════

    def plot_convergence(self, show: bool = True) -> None:
        """
        Plots the fitness convergence over generations.

        Shows:
            - Global best fitness (thick black line)
            - Per-island fitness (thin coloured lines)
            - Migration events as vertical dashed lines
        """
        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#F0F4F8')

        ax.set_title("Island GA — Fitness Convergence",
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Generation", fontsize=11)
        ax.set_ylabel("Best Fitness F(x)  [lower = better]", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        # ── global best ────────────────────────────────────────────────────
        global_hist = self.manager.get_global_fitness_history()
        generations = list(range(1, len(global_hist) + 1))

        ax.plot(generations, global_hist,
                color='#1A1A2E', linewidth=2.5,
                label='Global best', zorder=5)

        # ── per-island ─────────────────────────────────────────────────────
        for i in range(self.params.num_islands):
            island      = self.manager.get_island(i)
            island_hist = island.get_fitness_history()
            island_gens = list(range(1, len(island_hist) + 1))
            colour      = VAN_COLOURS[i % len(VAN_COLOURS)]

            ax.plot(island_gens, island_hist,
                    color=colour, linewidth=1.2,
                    alpha=0.6, linestyle='--',
                    label=f'Island {i}', zorder=4)

        # ── migration event markers ────────────────────────────────────────
        migration_log  = self.manager.get_migration_log()
        migration_gens = sorted(set(e['generation'] for e in migration_log))

        first = True
        for mg in migration_gens:
            if mg <= len(global_hist):
                ax.axvline(x=mg, color='#E07B39', linewidth=0.8,
                           alpha=0.5, linestyle=':',
                           label='Migration event' if first else '')
                first = False

        # ── best fitness annotation ────────────────────────────────────────
        best_fitness = min(global_hist) if global_hist else 0
        best_gen     = global_hist.index(best_fitness) + 1 if global_hist else 0

        ax.annotate(f'Best: {best_fitness:.2f}\n(gen {best_gen})',
                    xy=(best_gen, best_fitness),
                    xytext=(best_gen + max(1, len(global_hist) * 0.05),
                            best_fitness + (max(global_hist) - min(global_hist)) * 0.1
                            if len(global_hist) > 1 else best_fitness + 1),
                    fontsize=9, color='#1A1A2E',
                    arrowprops=dict(arrowstyle='->', color='#1A1A2E', lw=1),
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.8))

        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

        # parameter summary
        info = (f"Islands: {self.params.num_islands}  |  "
                f"Topology: {self.params.island_topology.value}  |  "
                f"Pop/island: {self.params.population_size}  |  "
                f"Migration: every {self.params.migration_interval} gens "
                f"({self.params.migration_strategy.value})")
        ax.text(0.01, 0.01, info, transform=ax.transAxes,
                fontsize=8.5, color='#333333',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', alpha=0.8))

        plt.tight_layout()
        if show:
            plt.show()

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 3 — TIME WINDOW CHART
    # ══════════════════════════════════════════════════════════════════════════

    def plot_time_windows(self, show: bool = True) -> None:
        """
        Horizontal bar chart showing each customer's time window
        and the actual arrival time.

        Green bar  = time window [earliest, latest]
        Blue dot   = actual arrival time (on time or early)
        Red dot    = actual arrival time (late — violation)
        """
        ncustomers = len(self.customers)
        fig, ax     = plt.subplots(figsize=(12, max(6, ncustomers * 0.45)))
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#F0F4F8')

        ax.set_title("VRPTW — Time Windows vs Actual Arrivals",
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Time", fontsize=11)
        ax.set_ylabel("Customer", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')

        # collect all stops from schedule
        stops_by_customer = {}
        for vehicle in self.schedule:
            for stop in vehicle['stops']:
                stops_by_customer[stop['customer_id']] = stop

        customer_ids = sorted(stops_by_customer.keys())
        y_positions  = {cid: i for i, cid in enumerate(customer_ids)}

        for cid in customer_ids:
            stop      = stops_by_customer[cid]
            y         = y_positions[cid]
            earliest, latest = stop['time_window']
            arrival   = stop['arrival_time']
            service   = stop['service_start']
            violation = stop['violation']

            # time window bar (green)
            ax.barh(y, latest - earliest, left=earliest,
                    height=0.5, color='#2ECC71', alpha=0.4,
                    edgecolor='#27AE60', linewidth=0.8)

            # arrival dot
            dot_colour = '#E74C3C' if violation > 0 else '#2E75B6'
            ax.scatter(arrival, y, color=dot_colour,
                       s=80, zorder=5, edgecolors='white', linewidths=1)

            # service start marker (when vehicle actually begins service)
            ax.scatter(service, y, color='#F39C12',
                       s=40, marker='|', zorder=6, linewidths=2)

            # violation annotation
            if violation > 0:
                ax.annotate(f'+{violation:.1f}',
                            xy=(arrival, y),
                            xytext=(6, 0),
                            textcoords='offset points',
                            fontsize=7.5, color='#E74C3C',
                            fontweight='bold')

            # customer label
            ax.text(-2, y, f'C{cid}', ha='right', va='center',
                    fontsize=8.5, fontweight='bold', color='#333333')

        ax.set_yticks([])
        ax.set_xlim(left=-5)

        # legend
        legend_elements = [
            mpatches.Patch(facecolor='#2ECC71', alpha=0.4,
                           edgecolor='#27AE60', label='Time window [e, l]'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#2E75B6', markersize=9,
                   label='Arrival (on time / early)'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#E74C3C', markersize=9,
                   label='Arrival (LATE — violation)'),
            Line2D([0], [0], marker='|', color='#F39C12',
                   markersize=10, linewidth=2,
                   label='Service start'),
        ]
        ax.legend(handles=legend_elements, loc='lower right',
                  fontsize=9, framealpha=0.9)

        plt.tight_layout()
        if show:
            plt.show()

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 4 — DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════

    def plot_dashboard(self, show: bool = True) -> None:
        """
        A combined summary dashboard with:
            - Key metrics panel
            - Per-vehicle distance bar chart
            - Per-vehicle customers served bar chart
            - Island parameter summary
        """
        fig = plt.figure(figsize=(14, 8))
        fig.patch.set_facecolor('#F8F9FA')
        fig.suptitle("VRPTW Island GA — Solution Dashboard",
                     fontsize=15, fontweight='bold', y=1.01)

        gs = gridspec.GridSpec(2, 3, figure=fig,
                               hspace=0.45, wspace=0.35)

        ax_metrics    = fig.add_subplot(gs[0, 0])
        ax_dist       = fig.add_subplot(gs[0, 1])
        axcustomers  = fig.add_subplot(gs[0, 2])
        axparams     = fig.add_subplot(gs[1, 0])
        ax_convergence = fig.add_subplot(gs[1, 1:])

        for ax in [ax_metrics, ax_dist, axcustomers,
                   axparams, ax_convergence]:
            ax.set_facecolor('#F0F4F8')

        # ── panel 1: key metrics ───────────────────────────────────────────
        ax_metrics.axis('off')
        total_dist  = sum(v['total_distance']  for v in self.schedule)
        total_viol  = sum(v['total_violation'] for v in self.schedule)
        total_cust  = sum(v['customers_served'] for v in self.schedule)
        n_gens      = self.manager.current_generation + 1
        n_migrations = len(self.manager.get_migration_log())

        metrics = [
            ("Fitness F(x)",       f"{self.solution.get_fitness():.4f}"),
            ("Total distance",     f"{total_dist:.2f}"),
            ("Time violations",    f"{total_viol:.4f}"),
            ("Customers served",   str(total_cust)),
            ("Vehicles used",      str(len(self.schedule))),
            ("Generations",        str(n_gens)),
            ("Migration events",   str(n_migrations)),
            ("Solve time",         f"{self.solver.solve_time:.2f}s"),
        ]

        ax_metrics.set_title("Key Metrics", fontsize=11,
                             fontweight='bold', pad=8)
        for i, (label, value) in enumerate(metrics):
            y = 0.95 - i * 0.115
            ax_metrics.text(0.05, y, label,
                            transform=ax_metrics.transAxes,
                            fontsize=9, color='#555555')
            ax_metrics.text(0.95, y, value,
                            transform=ax_metrics.transAxes,
                            fontsize=9.5, fontweight='bold',
                            color='#1A1A2E', ha='right')

        # ── panel 2: distance per van ──────────────────────────────────────
        van_labels = [f"Van {v['vehicle']}" for v in self.schedule]
        distances  = [v['total_distance']   for v in self.schedule]
        colours    = [VAN_COLOURS[i % len(VAN_COLOURS)]
                      for i in range(len(self.schedule))]

        bars = ax_dist.bar(van_labels, distances,
                           color=colours, edgecolor='white',
                           linewidth=1.2)
        ax_dist.set_title("Distance per Van", fontsize=11,
                          fontweight='bold', pad=8)
        ax_dist.set_ylabel("Distance", fontsize=9)
        ax_dist.tick_params(axis='x', labelsize=8.5)

        for bar, val in zip(bars, distances):
            ax_dist.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.5,
                         f'{val:.1f}', ha='center', va='bottom',
                         fontsize=8, fontweight='bold')

        # ── panel 3: customers per van ─────────────────────────────────────
        served = [v['customers_served'] for v in self.schedule]
        bars2  = axcustomers.bar(van_labels, served,
                                  color=colours, edgecolor='white',
                                  linewidth=1.2)
        axcustomers.set_title("Customers per Van", fontsize=11,
                               fontweight='bold', pad=8)
        axcustomers.set_ylabel("Customers", fontsize=9)
        axcustomers.set_ylim(0, max(served) + 2)
        axcustomers.tick_params(axis='x', labelsize=8.5)
        axcustomers.yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True))

        for bar, val in zip(bars2, served):
            axcustomers.text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + 0.1,
                              str(val), ha='center', va='bottom',
                              fontsize=9, fontweight='bold')

        # ── panel 4: parameter summary ─────────────────────────────────────
        axparams.axis('off')
        p = self.params
        param_lines = [
            ("Islands",        str(p.num_islands)),
            ("Topology",       p.island_topology.value),
            ("Pop / island",   str(p.population_size)),
            ("Max gens",       str(p.max_generations)),
            ("Crossover",      f"{p.crossover_strategy.value} @ {p.crossover_rate}"),
            ("Mutation",       f"{p.mutation_strategy.value} @ {p.mutation_rate}"),
            ("Local search",   f"2-opt @ {p.local_search_rate}"),
            ("Migration",      f"every {p.migration_interval}g | "
                               f"{p.migration_strategy.value}"),
            ("Weights",        f"w1={p.w1} w2={p.w2} w3={p.w3}"),
        ]

        axparams.set_title("GA Parameters", fontsize=11,
                            fontweight='bold', pad=8)
        for i, (label, value) in enumerate(param_lines):
            y = 0.95 - i * 0.1
            axparams.text(0.05, y, label,
                           transform=axparams.transAxes,
                           fontsize=8.5, color='#555555')
            axparams.text(0.95, y, value,
                           transform=axparams.transAxes,
                           fontsize=8.5, fontweight='bold',
                           color='#1A1A2E', ha='right')

        # ── panel 5: mini convergence ──────────────────────────────────────
        global_hist = self.manager.get_global_fitness_history()
        gens        = list(range(1, len(global_hist) + 1))

        for i in range(p.num_islands):
            ih     = self.manager.get_island(i).get_fitness_history()
            colour = VAN_COLOURS[i % len(VAN_COLOURS)]
            ax_convergence.plot(range(1, len(ih)+1), ih,
                                color=colour, linewidth=1,
                                alpha=0.5, linestyle='--',
                                label=f'Island {i}')

        ax_convergence.plot(gens, global_hist,
                            color='#1A1A2E', linewidth=2.2,
                            label='Global best', zorder=5)

        # migration lines
        mig_gens = sorted(set(
            e['generation'] for e in self.manager.get_migration_log()
        ))
        first = True
        for mg in mig_gens:
            if mg <= len(global_hist):
                ax_convergence.axvline(
                    x=mg, color='#E07B39', linewidth=0.7,
                    alpha=0.5, linestyle=':',
                    label='Migration' if first else '')
                first = False

        ax_convergence.set_title("Fitness Convergence", fontsize=11,
                                 fontweight='bold', pad=8)
        ax_convergence.set_xlabel("Generation", fontsize=9)
        ax_convergence.set_ylabel("Fitness", fontsize=9)
        ax_convergence.legend(fontsize=8, loc='upper right',
                              framealpha=0.9)
        ax_convergence.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        if show:
            plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.vrptw_solver import VRPTWSolver
    from src.ga_parameters import GAParameters
    from src.ga_parameters import (TopologyType, MigrationStrategy,
                     InitStrategy, CrossoverStrategy, MutationStrategy)

    # ── run the solver ─────────────────────────────────────────────────────
    problem_file = sys.argv[1] if len(sys.argv) > 1 else "data/C101_small.txt"

    params = GAParameters()
    params.population_size          = 50
    params.max_generations          = 200
    params.num_islands              = 4
    params.island_topology          = TopologyType.RING
    params.crossover_rate           = 0.85
    params.crossover_strategy       = CrossoverStrategy.OX
    params.mutation_rate            = 0.1
    params.mutation_strategy        = MutationStrategy.BOTH
    params.local_search_rate        = 0.3
    params.migration_rate           = 0.8
    params.migration_interval       = 10
    params.migration_size           = 2
    params.migration_strategy       = MigrationStrategy.BEST
    params.population_init_strategy = InitStrategy.SAVINGS_NN_RANDOM
    params.w1 = 1.0
    params.w2 = 10.0
    params.w3 = 5.0

    solver = VRPTWSolver()
    solver.load_problem(problem_file)
    solver.configure_parameters(params)
    solver.solve()
    solver.display_solution()

    # ── build visualiser ───────────────────────────────────────────────────
    print("\n[Visualiser] Generating charts...")
    vis = Visualiser(solver)

    # save all charts to output/
    vis.save_all("output")

    print("\n  Charts saved to output/:")
    print("    route_map.png     — 2D delivery route map")
    print("    convergence.png   — fitness per generation per island")
    print("    time_windows.png  — arrival times vs time windows")
    print("    dashboard.png     — full solution dashboard")
    print()

    # also display interactively
    vis.show_all()