#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySide6 desktop GUI for the synthetic-only SR-DP assignment runner.
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal, Slot
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QProgressBar,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - shown only when dependency is missing.
    raise SystemExit(
        "PySide6 is required for the GUI. Run: pip install -r requirements.txt"
    ) from exc

from srdp_full_safe_framework import run_protocol, save_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICIES = [
    "no_memory",
    "full_memory",
    "perfect_only",
    "partial_only",
    "collapse_resistant",
]
SUMMARY_COLUMNS = [
    "memory_policy",
    "signal",
    "auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "collapse_low_std_rate",
    "collapse_strict_rate",
    "mean_memory_size",
]


class StdoutProxy:
    def __init__(self, signal: Signal):
        self.signal = signal
        self.buffer = ""

    def write(self, text: str) -> None:
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.signal.emit(line)

    def flush(self) -> None:
        if self.buffer.strip():
            self.signal.emit(self.buffer.strip())
        self.buffer = ""


class ExperimentWorker(QObject):
    log = Signal(str)
    finished = Signal(object, object)
    failed = Signal(str)

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args

    @Slot()
    def run(self) -> None:
        try:
            output_dir = Path(str(self.args.out_prefix)).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            proxy = StdoutProxy(self.log)
            with redirect_stdout(proxy):
                all_rows, seed_results, summary_rows = run_protocol(self.args)
                proxy.flush()

            summary_path = f"{self.args.out_prefix}_summary.csv"
            seed_path = f"{self.args.out_prefix}_seed_results.csv"
            rows_path = f"{self.args.out_prefix}_all_rows.csv"

            save_csv(summary_path, summary_rows)
            save_csv(seed_path, seed_results)
            save_csv(rows_path, all_rows)

            self.log.emit("Saved CSV files:")
            self.log.emit(f"- {summary_path}")
            self.log.emit(f"- {seed_path}")
            self.log.emit(f"- {rows_path}")
            self.finished.emit(summary_rows, [summary_path, seed_path, rows_path])
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread: Optional[QThread] = None
        self.worker: Optional[ExperimentWorker] = None
        self.last_output_dir = PROJECT_ROOT / "outputs"

        self.setWindowTitle("SRDP Assignment GUI")
        self.resize(1180, 760)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        title = QLabel("Synthetic SR-DP Runner")
        title.setObjectName("Title")
        subtitle = QLabel("Configure, run, and export the assignment-safe synthetic experiment.")
        subtitle.setObjectName("Subtitle")
        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.build_controls())
        splitter.addWidget(self.build_results())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root_layout.addWidget(splitter, 1)

        self.setCentralWidget(root)
        self.apply_style()

    def build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(340)
        panel.setMaximumWidth(420)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 10, 0)
        layout.setSpacing(10)

        run_group = QGroupBox("Run Settings")
        form = QFormLayout(run_group)
        form.setLabelAlignment(Qt.AlignLeft)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100)
        self.epochs.setValue(6)
        form.addRow("Epochs", self.epochs)

        self.trials = QSpinBox()
        self.trials.setRange(1, 1000)
        self.trials.setValue(10)
        form.addRow("Trials/person", self.trials)

        self.memory_cap = QSpinBox()
        self.memory_cap.setRange(0, 100)
        self.memory_cap.setValue(3)
        form.addRow("Memory cap", self.memory_cap)

        self.seeds = QLineEdit("1 2 3 4 5 10 20 30 40 42")
        form.addRow("Seeds", self.seeds)

        layout.addWidget(run_group)

        model_group = QGroupBox("Model Parameters")
        model_form = QFormLayout(model_group)

        self.member_bonus = self.make_double_spin(0.16, -10.0, 10.0, 0.01)
        model_form.addRow("Member bonus", self.member_bonus)

        self.nonmember_bonus = self.make_double_spin(0.04, -10.0, 10.0, 0.01)
        model_form.addRow("Nonmember bonus", self.nonmember_bonus)

        self.shadow_scale = self.make_double_spin(1.0, 0.0, 10.0, 0.05)
        model_form.addRow("Shadow scale", self.shadow_scale)

        self.min_sigma = self.make_double_spin(0.10, 0.0, 10.0, 0.01)
        model_form.addRow("Min sigma", self.min_sigma)

        self.shadow_noise_rate = self.make_double_spin(0.35, 0.0, 1.0, 0.01)
        model_form.addRow("Shadow noise", self.shadow_noise_rate)

        layout.addWidget(model_group)

        policy_group = QGroupBox("Memory Policies")
        policy_layout = QGridLayout(policy_group)
        self.policy_checks: Dict[str, QCheckBox] = {}
        for index, policy in enumerate(DEFAULT_POLICIES):
            check = QCheckBox(policy)
            check.setChecked(True)
            self.policy_checks[policy] = check
            policy_layout.addWidget(check, index // 2, index % 2)
        layout.addWidget(policy_group)

        output_group = QGroupBox("Output")
        output_form = QFormLayout(output_group)

        self.output_dir = QLineEdit(str(PROJECT_ROOT / "outputs"))
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.choose_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir, 1)
        output_row.addWidget(browse_button)
        output_form.addRow("Folder", output_row)

        self.output_prefix = QLineEdit("srdp_gui")
        output_form.addRow("Prefix", self.output_prefix)

        layout.addWidget(output_group)

        self.run_button = QPushButton("Run Experiment")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self.start_experiment)
        layout.addWidget(self.run_button)

        self.open_output_button = QPushButton("Open Output Folder")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self.open_output_folder)
        layout.addWidget(self.open_output_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        layout.addStretch(1)
        return panel

    def build_results(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusLabel")
        layout.addWidget(self.status_label)

        self.table = QTableWidget(0, len(SUMMARY_COLUMNS))
        self.table.setHorizontalHeaderLabels(SUMMARY_COLUMNS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table, 3)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(divider)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Run logs will appear here.")
        layout.addWidget(self.log_view, 2)

        return panel

    @staticmethod
    def make_double_spin(value: float, minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def choose_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose output folder",
            self.output_dir.text() or str(PROJECT_ROOT),
        )
        if selected:
            self.output_dir.setText(selected)

    def open_output_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output_dir)))

    def parse_seeds(self) -> List[int]:
        parts = [p for p in re.split(r"[\s,;]+", self.seeds.text().strip()) if p]
        if not parts:
            raise ValueError("Enter at least one seed.")
        try:
            return [int(p) for p in parts]
        except ValueError as exc:
            raise ValueError("Seeds must be integers separated by spaces or commas.") from exc

    def selected_policies(self) -> List[str]:
        policies = [name for name, check in self.policy_checks.items() if check.isChecked()]
        if not policies:
            raise ValueError("Select at least one memory policy.")
        return policies

    def collect_args(self) -> argparse.Namespace:
        output_dir = Path(self.output_dir.text().strip() or str(PROJECT_ROOT / "outputs"))
        prefix = self.output_prefix.text().strip()
        if not prefix:
            raise ValueError("Output prefix cannot be empty.")
        self.last_output_dir = output_dir

        return argparse.Namespace(
            epochs=self.epochs.value(),
            trials=self.trials.value(),
            memory_cap=self.memory_cap.value(),
            member_bonus=self.member_bonus.value(),
            nonmember_bonus=self.nonmember_bonus.value(),
            shadow_scale=self.shadow_scale.value(),
            min_sigma=self.min_sigma.value(),
            shadow_noise_rate=self.shadow_noise_rate.value(),
            seeds=self.parse_seeds(),
            memory_policies=self.selected_policies(),
            out_prefix=str(output_dir / prefix),
        )

    @Slot()
    def start_experiment(self) -> None:
        try:
            args = self.collect_args()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return

        self.table.setRowCount(0)
        self.log_view.clear()
        self.append_log("Starting experiment...")
        self.status_label.setText("Running")
        self.run_button.setEnabled(False)
        self.open_output_button.setEnabled(False)
        self.progress.setRange(0, 0)

        self.thread = QThread(self)
        self.worker = ExperimentWorker(args)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.clear_thread_refs)
        self.thread.start()

    @Slot(str)
    def append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    @Slot(object, object)
    def on_finished(self, summary_rows: List[Dict[str, Any]], paths: List[str]) -> None:
        self.populate_table(summary_rows)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.run_button.setEnabled(True)
        self.open_output_button.setEnabled(True)
        self.status_label.setText(f"Finished: {len(summary_rows)} summary rows")
        self.append_log("Done.")

    @Slot(str)
    def on_failed(self, error: str) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.run_button.setEnabled(True)
        self.open_output_button.setEnabled(False)
        self.status_label.setText("Failed")
        self.append_log(error)
        QMessageBox.critical(self, "Run failed", error)

    @Slot()
    def clear_thread_refs(self) -> None:
        self.thread = None
        self.worker = None

    def populate_table(self, rows: List[Dict[str, Any]]) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for col_index, key in enumerate(SUMMARY_COLUMNS):
                item = QTableWidgetItem(str(row.get(key, "")))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row_index, col_index, item)
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override name.
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "Run in progress", "Wait for the current run to finish.")
            event.ignore()
            return
        event.accept()

    def apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 10pt;
                color: #1f2933;
                background: #f6f7f9;
            }
            QGroupBox {
                border: 1px solid #d2d7de;
                border-radius: 6px;
                margin-top: 10px;
                padding: 10px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #2f5d62;
                font-weight: 600;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget {
                border: 1px solid #c9d0d8;
                border-radius: 5px;
                background: #ffffff;
                padding: 5px;
            }
            QPushButton {
                border: 1px solid #b7c0ca;
                border-radius: 5px;
                background: #ffffff;
                padding: 7px 10px;
            }
            QPushButton:hover {
                background: #eef4f4;
            }
            QPushButton:disabled {
                color: #8a96a3;
                background: #eceff2;
            }
            QPushButton#PrimaryButton {
                background: #2f5d62;
                color: #ffffff;
                border-color: #2f5d62;
                font-weight: 600;
            }
            QPushButton#PrimaryButton:hover {
                background: #386f75;
            }
            QLabel#Title {
                font-size: 20pt;
                font-weight: 700;
                color: #173b3f;
                background: transparent;
            }
            QLabel#Subtitle, QLabel#StatusLabel {
                color: #52616f;
                background: transparent;
            }
            QHeaderView::section {
                background: #edf1f3;
                border: 1px solid #d2d7de;
                padding: 5px;
                font-weight: 600;
            }
            """
        )


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
