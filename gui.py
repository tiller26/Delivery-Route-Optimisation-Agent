import sys
import os
import math
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSlider, QComboBox, QSpinBox,
    QDoubleSpinBox, QFileDialog, QGroupBox, QSplitter, QFrame,
    QProgressBar, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src import (
    VRPTWSolver, GAParameters,
    CrossoverStrategy, MutationStrategy,
    TopologyType, MigrationStrategy, InitStrategy
)
from src.island_manager import IslandManager

# ── colour palette ────────────────────────────────────────────────────────────
VAN_COLOURS = [
    '#2E75B6','#1D9E75','#E07B39','#9B59B6',
    '#E74C3C','#1ABC9C','#F39C12','#2C3E50',
]
BG_DARK   = "#1E1E2E"
BG_PANEL  = "#2A2A3E"
BG_CARD   = "#313150"
ACCENT    = "#2E75B6"
ACCENT2   = "#1D9E75"
TEXT_MAIN = "#E8E8F0"
TEXT_SUB  = "#9999BB"
BTN_RUN   = "#1D9E75"
BTN_STOP  = "#E74C3C"
BTN_RESET = "#E07B39"


# ══════════════════════════════════════════════════════════════════════════════
# WORKER THREAD — runs the GA without freezing the GUI
# ══════════════════════════════════════════════════════════════════════════════

class GAWorker(QObject):
    generation_done = pyqtSignal(int, float, list, object)  # gen, global_best, island_bests, chromosome
    evolution_done  = pyqtSignal(object)                     # final chromosome
    error_occurred  = pyqtSignal(str)

    def __init__(self, solver: VRPTWSolver):
        super().__init__()
        self.solver   = solver
        self.running  = True

    def stop(self):
        self.running = False

    def run(self):
        try:
            manager  = self.solver.get_island_manager()
            params   = self.solver.params  
            max_gen  = params.max_generations

            for gen in range(max_gen):
                if not self.running:
                    break

                # evolve all islands one generation
                for island in manager.islands:
                    island.run_generation()

                # migration
                if manager.is_migration_due(gen):
                    manager.execute_migration(gen)

                # update global best
                manager.updateglobal_best()
                manager.global_history.append(manager.global_best.get_fitness())
                manager.current_generation = gen

                # collect per-island bests
                island_bests = [
                    manager.get_island(i).get_best_solution().get_fitness()
                    for i in range(params.num_islands)
                ]

                # Update solver active solution for map rendering
                self.solver.solution = manager.global_best

                # emit signal every generation
                self.generation_done.emit(
                    gen + 1,
                    manager.global_best.get_fitness(),
                    island_bests,
                    manager.global_best
                )

                # check convergence
                if manager.check_global_convergence():
                    break

            # Rebuild schedule and emit completion
            self.solver.solution = manager.global_best
            self.solver.schedule = self.solver.build_outputschedule()  
            self.evolution_done.emit(manager.global_best)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB CANVAS WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

class RouteCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 5), facecolor=BG_PANEL)
        super().__init__(self.fig)
        self.ax  = self.fig.add_subplot(111)
        self.style_ax()
        self.fig.tight_layout(pad=1.5)

    def style_ax(self):
        self.ax.set_facecolor(BG_DARK)
        self.ax.tick_params(colors=TEXT_SUB, labelsize=8)
        self.ax.xaxis.label.set_color(TEXT_SUB)
        self.ax.yaxis.label.set_color(TEXT_SUB)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(BG_CARD)
        self.ax.grid(True, alpha=0.15, linestyle='--', color=TEXT_SUB)
        self.ax.set_title("Route Map", color=TEXT_MAIN, fontsize=10, fontweight='bold', pad=8)

    def draw_routes(self, customers, depot, schedule):
        self.ax.clear()
        self.style_ax()

        if not depot or not customers or not schedule:
            self.draw_idle()
            return

        depot_x, depot_y = depot.get_location()

        # draw routes
        for v_idx, vehicle in enumerate(schedule):
            colour = VAN_COLOURS[v_idx % len(VAN_COLOURS)]
            route  = vehicle['route']
            xs = [depot_x]
            ys = [depot_y]
            for cid in route:
                cx, cy = customers[cid].get_location()
                xs.append(cx)
                ys.append(cy)
            xs.append(depot_x)
            ys.append(depot_y)

            self.ax.plot(xs, ys, '-', color=colour, linewidth=1.8, alpha=0.75, zorder=2)

            # arrows
            for i in range(len(xs) - 1):
                dx  = xs[i+1] - xs[i]
                dy  = ys[i+1] - ys[i]
                mx  = xs[i] + dx * 0.5
                my  = ys[i] + dy * 0.5
                self.ax.annotate(
                    '', xy=(mx + dx*0.01, my + dy*0.01),
                    xytext=(mx - dx*0.01, my - dy*0.01),
                    arrowprops=dict(arrowstyle='->', color=colour, lw=1.4),
                    zorder=3
                )

        # customers
        for cid, customer in customers.items():
            cx, cy = customer.get_location()
            c_col  = '#AAAACC'
            for v_idx, vehicle in enumerate(schedule):
                if cid in vehicle['route']:
                    c_col = VAN_COLOURS[v_idx % len(VAN_COLOURS)]
                    break
            self.ax.scatter(cx, cy, s=100, color=c_col, edgecolors='white', linewidths=1.2, zorder=4)
            self.ax.annotate(str(cid), xy=(cx, cy), xytext=(4, 4), textcoords='offset points',
                             fontsize=7.5, color='white', fontweight='bold', zorder=5)

        # depot
        self.ax.scatter(depot_x, depot_y, s=250, color='#FFD700', marker='*', zorder=6)
        self.ax.annotate('DEPOT', xy=(depot_x, depot_y), xytext=(5, 5), textcoords='offset points',
                         fontsize=8, color='#FFD700', fontweight='bold', zorder=7)
        self.draw_idle()

    def drawcustomers_only(self, customers, depot):
        self.ax.clear()
        self.style_ax()

        if not depot:
            self.draw_idle()
            return

        depot_x, depot_y = depot.get_location()

        for cid, customer in customers.items():
            cx, cy = customer.get_location()
            self.ax.scatter(cx, cy, s=100, color=ACCENT, edgecolors='white', linewidths=1.2, zorder=4)
            self.ax.annotate(str(cid), xy=(cx, cy), xytext=(4, 4), textcoords='offset points',
                             fontsize=7.5, color='white', fontweight='bold', zorder=5)

        self.ax.scatter(depot_x, depot_y, s=250, color='#FFD700', marker='*', zorder=6)
        self.ax.annotate('DEPOT', xy=(depot_x, depot_y), xytext=(5, 5), textcoords='offset points',
                         fontsize=8, color='#FFD700', fontweight='bold', zorder=7)

        self.ax.set_title("Route Map — Load & Run to See Routes", color=TEXT_MAIN, fontsize=10, fontweight='bold', pad=8)
        self.draw_idle()


class ConvergenceCanvas(FigureCanvas):
    def __init__(self, num_islands=4):
        self.fig = Figure(figsize=(5, 3), facecolor=BG_PANEL)
        super().__init__(self.fig)
        self.ax          = self.fig.add_subplot(111)
        self.num_islands = num_islands
        self.global_hist = []
        self.island_hist = [[] for _ in range(num_islands)]
        self.style_ax()
        self.fig.tight_layout(pad=1.5)

    def style_ax(self):
        self.ax.set_facecolor(BG_DARK)
        self.ax.tick_params(colors=TEXT_SUB, labelsize=8)
        self.ax.xaxis.label.set_color(TEXT_SUB)
        self.ax.yaxis.label.set_color(TEXT_SUB)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(BG_CARD)
        self.ax.grid(True, alpha=0.15, linestyle='--', color=TEXT_SUB)
        self.ax.set_title("Fitness Convergence", color=TEXT_MAIN, fontsize=10, fontweight='bold', pad=8)
        self.ax.set_xlabel("Generation", color=TEXT_SUB, fontsize=8)
        self.ax.set_ylabel("Best Fitness", color=TEXT_SUB, fontsize=8)

    def reset(self, num_islands):
        self.num_islands = num_islands
        self.global_hist = []
        self.island_hist = [[] for _ in range(num_islands)]
        self.ax.clear()
        self.style_ax()
        self.draw_idle()

    def update_chart(self, gen: int, global_best: float, island_bests: list):
        self.global_hist.append(global_best)
        for i, val in enumerate(island_bests):
            if i < len(self.island_hist):
                self.island_hist[i].append(val)

        self.ax.clear()
        self.style_ax()

        gens = list(range(1, len(self.global_hist) + 1))

        # island lines
        for i in range(self.num_islands):
            if i < len(self.island_hist) and self.island_hist[i]:
                self.ax.plot(
                    range(1, len(self.island_hist[i]) + 1),
                    self.island_hist[i],
                    color=VAN_COLOURS[i % len(VAN_COLOURS)],
                    linewidth=1, alpha=0.55, linestyle='--', label=f'Island {i}'
                )

        # global best
        if self.global_hist:
            self.ax.plot(gens, self.global_hist, color='#FFD700', linewidth=2.2, label='Global best', zorder=5)

        self.ax.legend(fontsize=7, loc='upper right', facecolor=BG_CARD, labelcolor=TEXT_MAIN, framealpha=0.8)
        self.draw_idle()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════

class VRPTWGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VRPTW Island GA — Intelligent Delivery Optimiser")
        self.setMinimumSize(1300, 800)

        self.solver      = VRPTWSolver()
        self.thread      = None
        self.worker      = None
        self.problem_loaded = False

        self.apply_dark_theme()
        self.build_ui()

    def apply_dark_theme(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color: {BG_DARK}; color: {TEXT_MAIN}; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; }}
            QGroupBox {{ background-color: {BG_PANEL}; border: 1px solid {BG_CARD}; border-radius: 8px; margin-top: 10px; padding-top: 8px; font-weight: bold; color: {TEXT_MAIN}; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 6px; color: {ACCENT}; font-size: 12px; font-weight: bold; }}
            QPushButton {{ background-color: {BG_CARD}; color: {TEXT_MAIN}; border: 1px solid {ACCENT}; border-radius: 6px; padding: 7px 14px; font-weight: bold; }}
            QPushButton:hover {{ background-color: {ACCENT}; }}
            QPushButton:disabled {{ background-color: {BG_PANEL}; color: {TEXT_SUB}; border-color: {BG_CARD}; }}
            QPushButton#run_btn {{ background-color: {BTN_RUN}; border-color: {BTN_RUN}; font-size: 14px; }}
            QPushButton#run_btn:hover {{ background-color: #17845F; }}
            QPushButton#stop_btn {{ background-color: {BTN_STOP}; border-color: {BTN_STOP}; }}
            QPushButton#reset_btn {{ background-color: {BTN_RESET}; border-color: {BTN_RESET}; }}
            QSlider::groove:horizontal {{ height: 4px; background: {BG_CARD}; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
            QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
            QComboBox {{ background-color: {BG_CARD}; border: 1px solid {ACCENT}; border-radius: 5px; padding: 4px 8px; color: {TEXT_MAIN}; }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{ background-color: {BG_CARD}; color: {TEXT_MAIN}; selection-background-color: {ACCENT}; }}
            QSpinBox, QDoubleSpinBox {{ background-color: {BG_CARD}; border: 1px solid {ACCENT}; border-radius: 5px; padding: 3px 6px; color: {TEXT_MAIN}; }}
            QLabel#metric_val {{ color: {ACCENT2}; font-size: 18px; font-weight: bold; }}
            QLabel#metric_lbl {{ color: {TEXT_SUB}; font-size: 11px; }}
            QProgressBar {{ background-color: {BG_CARD}; border: 1px solid {BG_CARD}; border-radius: 4px; height: 8px; text-align: center; color: transparent; }}
            QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 4px; }}
        """)

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root    = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        left   = self.build_left_panel()
        centre = self.build_centre_panel()
        right  = self.build_right_panel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(centre)
        splitter.addWidget(right)
        splitter.setSizes([280, 620, 380])

        root.addWidget(splitter)

    def build_left_panel(self) -> QWidget:
        panel = QScrollArea()
        panel.setWidgetResizable(True)
        panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel.setStyleSheet("QScrollArea { border: none; }")

        inner  = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        # ── Problem Instance Loader ───────────────────────────────────────────
        file_group = QGroupBox("Problem Instance")
        file_lay   = QVBoxLayout(file_group)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet(f"color: {TEXT_SUB}; font-size: 11px;")

        load_btn = QPushButton("  Browse Solomon File")
        load_btn.clicked.connect(self.load_file)

        self.problem_info = QLabel("")
        self.problem_info.setStyleSheet(f"color: {ACCENT2}; font-size: 11px;")

        file_lay.addWidget(self.file_label)
        file_lay.addWidget(load_btn)
        file_lay.addWidget(self.problem_info)
        layout.addWidget(file_group)

        # ── GA parameters ─────────────────────────────────────────────────────
        ga_group = QGroupBox("GA Parameters")
        ga_lay   = QGridLayout(ga_group)
        ga_lay.setSpacing(6)
        row = 0

        def add_spin(label, min_v, max_v, default, step=1, is_float=False):
            nonlocal row
            ga_lay.addWidget(QLabel(label), row, 0)
            sb = QDoubleSpinBox() if is_float else QSpinBox()
            if is_float:
                sb.setDecimals(2)
            sb.setSingleStep(step)
            sb.setMinimum(min_v)
            sb.setMaximum(max_v)
            sb.setValue(default)
            ga_lay.addWidget(sb, row, 1)
            row += 1
            return sb

        def add_combo(label, options):
            nonlocal row
            ga_lay.addWidget(QLabel(label), row, 0)
            cb = QComboBox()
            cb.addItems(options)
            ga_lay.addWidget(cb, row, 1)
            row += 1
            return cb

        self.pop_size     = add_spin("Population size",  10, 500, 50, 10)
        self.max_gens     = add_spin("Max generations",   10, 2000, 200, 50)
        self.cx_rate      = add_spin("Crossover rate",  0.0, 1.0, 0.85, 0.05, True)
        self.cx_strategy  = add_combo("Crossover strategy", ["OX", "PMX"])
        self.mut_rate     = add_spin("Mutation rate",   0.0, 1.0, 0.10, 0.01, True)
        self.mut_strategy = add_combo("Mutation strategy", ["SWAP", "INVERSION", "BOTH"])
        self.ls_rate      = add_spin("2-opt rate",      0.0, 1.0, 0.30, 0.05, True)
        
        # CHANGED: Added dynamic Init Strategy combo box selection to main parameter options
        self.init_strategy = add_combo("Initialization strategy", ["SAVINGS_NN_RANDOM", "NN_RANDOM", "RANDOM_ONLY"])

        layout.addWidget(ga_group)

        # ── Island Parameters ─────────────────────────────────────────────────
        isl_group = QGroupBox("Island Parameters")
        isl_lay   = QGridLayout(isl_group)
        isl_lay.setSpacing(6)
        row = 0

        def add_isl_spin(label, min_v, max_v, default, step=1, is_float=False):
            nonlocal row
            isl_lay.addWidget(QLabel(label), row, 0)
            sb = QDoubleSpinBox() if is_float else QSpinBox()
            if is_float:
                sb.setDecimals(2)
            sb.setSingleStep(step)
            sb.setMinimum(min_v)
            sb.setMaximum(max_v)
            sb.setValue(default)
            isl_lay.addWidget(sb, row, 1)
            row += 1
            return sb

        def add_isl_combo(label, options):
            nonlocal row
            isl_lay.addWidget(QLabel(label), row, 0)
            cb = QComboBox()
            cb.addItems(options)
            isl_lay.addWidget(cb, row, 1)
            row += 1
            return cb

        self.ui_num_islands = add_isl_spin("Num islands",       2, 16,  4)
        self._topology       = add_isl_combo("Topology", ["RING", "FULLY_CONNECTED", "STAR", "RANDOM"])
        self.mig_rate       = add_isl_spin("Migration rate",  0.0, 1.0, 0.8, 0.05, True)
        self.mig_interval   = add_isl_spin("Migration interval", 1, 100, 10)
        self.mig_size       = add_isl_spin("Migration size",     1,  20,  2)
        self.mig_strategy   = add_isl_combo("Migration strategy", ["BEST", "RANDOM", "MIXED"])

        layout.addWidget(isl_group)

        # ── Fitness Weight Balancers ──────────────────────────────────────────
        fw_group = QGroupBox("Fitness Weights")
        fw_lay   = QGridLayout(fw_group)
        fw_lay.setSpacing(6)
        row = 0

        def add_fw(label, default):
            nonlocal row
            fw_lay.addWidget(QLabel(label), row, 0)
            sb = QDoubleSpinBox()
            sb.setDecimals(1)
            sb.setSingleStep(0.5)
            sb.setMinimum(0.0)
            sb.setMaximum(100.0)
            sb.setValue(default)
            fw_lay.addWidget(sb, row, 1)
            row += 1
            return sb

        self.w1 = add_fw("w1 — Distance",   1.0)
        self.w2 = add_fw("w2 — Violations", 10.0)
        self.w3 = add_fw("w3 — Vehicles",   5.0)
        layout.addWidget(fw_group)

        btn_group = QGroupBox("Controls")
        btn_lay   = QVBoxLayout(btn_group)
        btn_lay.setSpacing(8)

        self.run_btn   = QPushButton("▶  Run Optimisation")
        self.run_btn.setObjectName("run_btn")
        self.run_btn.setMinimumHeight(42)
        self.run_btn.clicked.connect(self.run)
        self.run_btn.setEnabled(False)

        self.stop_btn  = QPushButton("■  Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)

        self.reset_btn = QPushButton("↺  Reset")
        self.reset_btn.setObjectName("reset_btn")
        self.reset_btn.clicked.connect(self.reset)

        self.save_btn  = QPushButton("💾  Save Charts & CSV")
        self.save_btn.clicked.connect(self.save)
        self.save_btn.setEnabled(False)

        btn_lay.addWidget(self.run_btn)
        btn_lay.addWidget(self.stop_btn)
        btn_lay.addWidget(self.reset_btn)
        btn_lay.addWidget(self.save_btn)
        layout.addWidget(btn_group)

        layout.addStretch()
        panel.setWidget(inner)
        return panel

    def build_centre_panel(self) -> QWidget:
        panel  = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel("Live Route Map")
        title.setStyleSheet(f"color: {TEXT_MAIN}; font-size: 15px; font-weight: bold; padding: 4px 0;")
        layout.addWidget(title)

        self.route_canvas = RouteCanvas()
        layout.addWidget(self.route_canvas)

        self.progress = QProgressBar()
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(8)
        layout.addWidget(self.progress)

        self.status_label = QLabel("Load a problem file to begin.")
        self.status_label.setStyleSheet(f"color: {TEXT_SUB}; font-size: 11px; padding: 2px 0;")
        layout.addWidget(self.status_label)

        return panel

    def build_right_panel(self) -> QWidget:
        panel  = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        metrics_group = QGroupBox("Live Metrics")
        metrics_lay   = QGridLayout(metrics_group)
        metrics_lay.setSpacing(8)

        def add_metric(label, row, col):
            lbl = QLabel(label)
            lbl.setObjectName("metric_lbl")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val = QLabel("—")
            val.setObjectName("metric_val")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            metrics_lay.addWidget(lbl, row*2+1, col)
            metrics_lay.addWidget(val, row*2,   col)
            return val

        self.m_generation = add_metric("Generation",    0, 0)
        self.m_fitness    = add_metric("Best Fitness",  0, 1)
        self.m_vehicles   = add_metric("Vehicles",      1, 0)
        self.m_violations = add_metric("Violations",    1, 1)
        self.m_distance   = add_metric("Distance",      2, 0)
        self.m_migrations = add_metric("Migrations",    2, 1)

        layout.addWidget(metrics_group)

        chart_title = QLabel("Convergence Chart")
        chart_title.setStyleSheet(f"color: {TEXT_MAIN}; font-size: 13px; font-weight: bold; padding: 2px 0;")
        layout.addWidget(chart_title)

        self.conv_canvas = ConvergenceCanvas(num_islands=4)
        layout.addWidget(self.conv_canvas)

        van_title = QLabel("Route Summary")
        van_title.setStyleSheet(f"color: {TEXT_MAIN}; font-size: 13px; font-weight: bold; padding: 2px 0;")
        layout.addWidget(van_title)

        self.van_summary = QLabel("—")
        self.van_summary.setWordWrap(True)
        self.van_summary.setStyleSheet(f"color: {TEXT_SUB}; font-size: 11px; background: {BG_PANEL}; border-radius: 6px; padding: 8px;")
        layout.addWidget(self.van_summary)

        return panel

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Solomon Instance", "data", "Text Files (*.txt);;All Files (*)")
        if not path:
            return

        try:
            self.solver = VRPTWSolver()
            self.solver.load_problem(path)
            self.problem_loaded = True

            name = os.path.basename(path)
            n    = len(self.solver.get_customers())
            self.file_label.setText(name)
            self.problem_info.setText(f"{n} customers | {self.solver.vehicle_count} vehicles | cap {self.solver.vehicle_cap:.0f}")

            self.route_canvas.drawcustomers_only(self.solver.get_customers(), self.solver.get_depot())
            self.run_btn.setEnabled(True)
            self.set_status(f"Loaded: {name}  ({n} customers)")

        except Exception as e:
            self.set_status(f"Error loading file: {e}", error=True)

    def buildparams(self) -> GAParameters:
        """Reads all GUI widgets and returns a GAParameters object."""
        p = GAParameters()
        p.population_size    = self.pop_size.value()
        p.max_generations    = self.max_gens.value()
        p.crossover_rate     = self.cx_rate.value()
        p.crossover_strategy = (CrossoverStrategy.OX if self.cx_strategy.currentText() == "OX" else CrossoverStrategy.PMX)
        p.mutation_rate      = self.mut_rate.value()
        p.mutation_strategy  = {
            "SWAP":      MutationStrategy.SWAP,
            "INVERSION": MutationStrategy.INVERSION,
            "BOTH":      MutationStrategy.BOTH,
        }[self.mut_strategy.currentText()]
        p.local_search_rate  = self.ls_rate.value()

        p.num_islands        = self.ui_num_islands.value()
        p.island_topology    = {
            "RING":            TopologyType.RING,
            "FULLY_CONNECTED": TopologyType.FULLY_CONNECTED,
            "STAR":            TopologyType.STAR,
            "RANDOM":          TopologyType.RANDOM,
        }[self._topology.currentText()]
        p.migration_rate     = self.mig_rate.value()
        p.migration_interval = self.mig_interval.value()
        p.migration_size     = self.mig_size.value()
        p.migration_strategy = {
            "BEST":   MigrationStrategy.BEST,
            "RANDOM": MigrationStrategy.RANDOM,
            "MIXED":  MigrationStrategy.MIXED,
        }[self.mig_strategy.currentText()]

        # CHANGED: Map dropdown selections dynamically to InitStrategy choices for parameter study configuration
        p.population_init_strategy = {
            "SAVINGS_NN_RANDOM": InitStrategy.SAVINGS_NN_RANDOM,
            "NN_RANDOM":        InitStrategy.NN_RANDOM,
            "RANDOM_ONLY":      InitStrategy.RANDOM_ONLY
        }[self.init_strategy.currentText()]

        p.w1 = self.w1.value()
        p.w2 = self.w2.value()
        p.w3 = self.w3.value()
        return p

    def run(self):
        if not self.problem_loaded:
            self.set_status("Load a problem file first.", error=True)
            return

        params = self.buildparams()
        try:
            params.validate()
        except ValueError as e:
            self.set_status(str(e), error=True)
            return

        self.solver.configure_parameters(params)

        from src.fitness_evaluator import FitnessEvaluator
        self.solver.evaluator = FitnessEvaluator(  
            customers  = self.solver.get_customers(),
            depot      = self.solver.get_depot(),
            parameters = params
        )
        self.solver.island_manager = IslandManager(params = params, evaluator = self.solver.evaluator)  
        self.solver.island_manager.initialise_all_islands(customers = self.solver.get_customers(), depot = self.solver.get_depot())

        self.conv_canvas.reset(params.num_islands)
        self.progress.setMaximum(params.max_generations)
        self.progress.setValue(0)

        self.thread = QThread()
        self.worker = GAWorker(self.solver)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.generation_done.connect(self.on_generation)
        self.worker.evolution_done.connect(self.on_done)
        self.worker.error_occurred.connect(self.on_error)
        self.thread.start()

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.set_status("Optimising...")

    def stop(self):
        if self.worker:
            self.worker.stop()
        self.set_status("Stopped by user.")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def reset(self):
        if self.worker:
            self.worker.stop()
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        self.solver       = VRPTWSolver()
        self.problem_loaded = False

        self.route_canvas.ax.clear()
        self.route_canvas.style_ax()
        self.route_canvas.draw_idle()

        self.conv_canvas.reset(self.ui_num_islands.value())

        for lbl in [self.m_generation, self.m_fitness, self.m_vehicles, self.m_violations, self.m_distance, self.m_migrations]:
            lbl.setText("—")

        self.van_summary.setText("—")
        self.progress.setValue(0)
        self.file_label.setText("No file loaded")
        self.problem_info.setText("")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.set_status("Reset. Load a problem file to begin.")

    def save(self):
        os.makedirs("output", exist_ok=True)
        try:
            self.solver.export_solution("output/solution.csv")
            self.solver.export_fitness_history("output/fitness_history.csv")
            self.set_status("Saved solution logs safely to output/ directory.")
        except Exception as e:
            self.set_status(f"Save error: {e}", error=True)

    def on_generation(self, gen: int, global_best: float, island_bests: list, chromosome):
        self.conv_canvas.update_chart(gen, global_best, island_bests)
        self.progress.setValue(gen)
        self.m_generation.setText(str(gen))
        self.m_fitness.setText(f"{global_best:.2f}")

        if gen % 5 == 0:
            schedule = self.solver.build_outputschedule()  
            self.route_canvas.draw_routes(self.solver.get_customers(), self.solver.get_depot(), schedule)
            total_viol = sum(v['total_violation'] for v in schedule)
            total_dist = sum(v['total_distance']  for v in schedule)
            self.m_vehicles.setText(str(len(schedule)))
            self.m_violations.setText(f"{total_viol:.3f}")
            self.m_distance.setText(f"{total_dist:.1f}")

        mig_count = len(self.solver.get_island_manager().get_migration_log())
        self.m_migrations.setText(str(mig_count))
        self.set_status(f"Generation {gen}  | Best fitness: {global_best:.4f}")

    def on_done(self, chromosome):
        if self.thread:
            self.thread.quit()
            self.thread.wait()

        schedule = self.solver.get_schedule()
        self.route_canvas.draw_routes(self.solver.get_customers(), self.solver.get_depot(), schedule)

        total_dist = sum(v['total_distance']  for v in schedule)
        total_viol = sum(v['total_violation'] for v in schedule)
        self.m_fitness.setText(f"{chromosome.get_fitness():.4f}")
        self.m_vehicles.setText(str(len(schedule)))
        self.m_violations.setText(f"{total_viol:.4f}")
        self.m_distance.setText(f"{total_dist:.2f}")
        self.progress.setValue(self.progress.maximum())

        summary_lines = []
        for v in schedule:
            route_str = " → ".join(str(c) for c in v['route'])
            summary_lines.append(
                f"<b style='color:{VAN_COLOURS[(v['vehicle']-1) % len(VAN_COLOURS)]}'>Van {v['vehicle']}</b>: "
                f"{route_str}  <span style='color:{TEXT_SUB}'>({v['customers_served']} stops, dist {v['total_distance']:.1f})</span>"
            )
        self.van_summary.setText("<br>".join(summary_lines))

        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.set_status(f"Done | Fitness: {chromosome.get_fitness():.4f} | Vans: {len(schedule)}")

    def on_error(self, msg: str):
        self.set_status(f"Error: {msg}", error=True)
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def set_status(self, msg: str, error: bool = False):
        colour = "#E74C3C" if error else TEXT_SUB
        self.status_label.setStyleSheet(f"color: {colour}; font-size: 11px; padding: 2px 0;")
        self.status_label.setText(msg)

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = VRPTWGui()
    win.show()
    sys.exit(app.exec())