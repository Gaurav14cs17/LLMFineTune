"""
Full Fine-tuning Tab - Run torchtune full finetune recipes via the UI.

Supports single-device and distributed (multi-GPU) full fine-tuning of LLMs
using `tune run` under the hood.
"""

import os
import sys
import subprocess
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QProgressBar, QFileDialog, QMessageBox,
    QGridLayout, QFrame, QScrollArea, QPlainTextEdit, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

import torch

from ui.styles import (
    BTN_DANGER, BTN_PRIMARY_LARGE, BTN_SECONDARY,
    CHECK_STYLE, COMBO_STYLE, EDIT_STYLE, LABEL_HEADING,
    LABEL_SECONDARY, LOG_STYLE, PROGRESS_STYLE, SPIN_STYLE,
)
from ui.helpers import get_project_root, list_recipes, list_configs_for_recipe


class TrainingWorker(QThread):
    log_signal = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, cmd, cwd):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self.process = None
        self._running = True

    def run(self):
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.cwd,
                bufsize=1,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}
            )
            while self._running:
                line = self.process.stdout.readline()
                if not line:
                    if self.process.poll() is not None:
                        break
                    continue
                self.log_signal.emit(line.rstrip())
            if self.process:
                self.process.wait(timeout=5)
                self.finished.emit(self.process.returncode or 0)
            else:
                self.finished.emit(-1)
        except Exception as e:
            self.log_signal.emit(f"ERROR: {str(e)}")
            self.finished.emit(-1)

    def stop(self):
        self._running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self.process.kill()
                except OSError:
                    pass


def _gpu_info() -> list:
    try:
        gpus = []
        if not torch.cuda.is_available():
            return gpus
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            try:
                total = torch.cuda.get_device_properties(i).total_mem / 1e6
                free = (torch.cuda.get_device_properties(i).total_mem
                        - torch.cuda.memory_reserved(i)) / 1e6
            except Exception:
                total = free = 0
            gpus.append({"idx": i, "name": name, "total_mb": total, "free_mb": free})
        return gpus
    except Exception:
        return []


FULL_FT_RECIPES = [
    "full_finetune_single_device",
    "full_finetune_distributed",
]


class TrainingTab(QWidget):
    training_started = pyqtSignal()
    training_stopped = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.worker = None
        self.project_root = get_project_root()
        self._all_recipes = list_recipes()
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Vertical)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        config_w = QWidget()
        cl = QVBoxLayout(config_w)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(4)

        cl.addWidget(self._build_info_banner())

        row1 = QHBoxLayout()
        row1.addWidget(self._build_recipe_group(), 5)
        row1.addWidget(self._build_device_group(), 3)
        cl.addLayout(row1)

        cl.addWidget(self._build_overrides_group())
        cl.addWidget(self._build_summary_card())
        cl.addLayout(self._build_buttons())
        cl.addLayout(self._build_progress())

        scroll.setWidget(config_w)
        splitter.addWidget(scroll)
        splitter.addWidget(self._build_log())
        splitter.setSizes([440, 280])
        main_layout.addWidget(splitter)

        self._update_summary()

    def _build_info_banner(self):
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame{background:#e3f2fd;border:1px solid #90caf9;"
            "border-radius:6px;padding:10px 14px;}")
        lay = QVBoxLayout(frame)
        lay.setSpacing(4)
        title = QLabel("Full Fine-tuning")
        title.setStyleSheet("font-size:14px;font-weight:700;color:#1565c0;")
        lay.addWidget(title)
        desc = QLabel(
            "Train all model parameters end-to-end using torchtune recipes. "
            "This requires the most GPU memory but provides maximum flexibility.\n\n"
            "Workflow: Select a recipe + config, optionally override parameters, "
            "then click Start. The UI runs `tune run <recipe> --config <config>` "
            "under the hood.")
        desc.setStyleSheet("font-size:12px;color:#0d47a1;")
        desc.setWordWrap(True)
        lay.addWidget(desc)
        return frame

    def _build_recipe_group(self):
        g = QGroupBox("Recipe & Configuration")
        lay = QVBoxLayout(g)

        recipe_lay = QHBoxLayout()
        recipe_lay.addWidget(QLabel("Recipe:"))
        self.recipe_combo = QComboBox()
        self.recipe_combo.setStyleSheet(COMBO_STYLE)
        self.recipe_combo.setMinimumWidth(280)
        for r in self._all_recipes:
            if any(ft in r["name"] for ft in ("full_finetune", "full_finetune_distributed")):
                label = r["name"]
                if r["supports_distributed"]:
                    label += "  (multi-GPU)"
                self.recipe_combo.addItem(label, r["name"])
        self.recipe_combo.currentIndexChanged.connect(self._on_recipe_changed)
        recipe_lay.addWidget(self.recipe_combo)
        recipe_lay.addStretch()
        lay.addLayout(recipe_lay)

        config_lay = QHBoxLayout()
        config_lay.addWidget(QLabel("Config:"))
        self.config_combo = QComboBox()
        self.config_combo.setStyleSheet(COMBO_STYLE)
        self.config_combo.setMinimumWidth(350)
        self.config_combo.currentTextChanged.connect(lambda _: self._update_summary())
        config_lay.addWidget(self.config_combo)
        config_lay.addStretch()
        lay.addLayout(config_lay)

        self._on_recipe_changed()

        ckpt_lay = QHBoxLayout()
        ckpt_lay.addWidget(QLabel("Checkpoint Dir:"))
        self.ckpt_dir_edit = QLineEdit()
        self.ckpt_dir_edit.setPlaceholderText("Path to downloaded model checkpoints")
        self.ckpt_dir_edit.setStyleSheet(EDIT_STYLE)
        ckpt_lay.addWidget(self.ckpt_dir_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.setStyleSheet(BTN_SECONDARY)
        browse_btn.clicked.connect(self._browse_ckpt_dir)
        ckpt_lay.addWidget(browse_btn)
        lay.addLayout(ckpt_lay)

        output_lay = QHBoxLayout()
        output_lay.addWidget(QLabel("Output Dir:"))
        self.output_dir_edit = QLineEdit("output/full_finetune")
        self.output_dir_edit.setStyleSheet(EDIT_STYLE)
        output_lay.addWidget(self.output_dir_edit)
        out_browse = QPushButton("Browse")
        out_browse.setFixedWidth(80)
        out_browse.setStyleSheet(BTN_SECONDARY)
        out_browse.clicked.connect(lambda: self._browse_dir(self.output_dir_edit))
        output_lay.addWidget(out_browse)
        lay.addLayout(output_lay)

        return g

    def _build_device_group(self):
        g = QGroupBox("Device & Hardware")
        lay = QVBoxLayout(g)

        dev_lay = QHBoxLayout()
        dev_lay.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU")
        gpus = _gpu_info()
        for gi in gpus:
            self.device_combo.addItem(f"GPU {gi['idx']}: {gi['name'][:25]}")
        if gpus:
            self.device_combo.setCurrentIndex(1)
        self.device_combo.setStyleSheet(COMBO_STYLE)
        dev_lay.addWidget(self.device_combo)
        lay.addLayout(dev_lay)

        self.gpu_info_frame = QFrame()
        self.gpu_info_frame.setStyleSheet(
            "QFrame{background:#e8edf3;border:1px solid #c9cdd3;border-radius:4px;padding:8px}")
        gif_lay = QVBoxLayout(self.gpu_info_frame)
        gif_lay.setSpacing(2)
        self.gpu_name_label = QLabel("")
        self.gpu_name_label.setStyleSheet("font-weight:600;color:#394867;font-size:13px;")
        self.gpu_mem_label = QLabel("")
        self.gpu_mem_label.setStyleSheet("color:#212d40;font-size:13px;")
        gif_lay.addWidget(self.gpu_name_label)
        gif_lay.addWidget(self.gpu_mem_label)
        lay.addWidget(self.gpu_info_frame)
        self._update_gpu_info()
        self.device_combo.currentIndexChanged.connect(lambda _: self._update_gpu_info())

        nproc_lay = QHBoxLayout()
        nproc_lay.addWidget(QLabel("Num GPUs:"))
        self.nproc_spin = QSpinBox()
        self.nproc_spin.setRange(1, 8)
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.nproc_spin.setValue(min(n_gpu, 1))
        self.nproc_spin.setStyleSheet(SPIN_STYLE)
        self.nproc_spin.setToolTip("Number of GPUs for distributed training (torchrun --nproc_per_node)")
        nproc_lay.addWidget(self.nproc_spin)
        nproc_lay.addStretch()
        lay.addLayout(nproc_lay)

        self.act_ckpt_check = QCheckBox("Activation Checkpointing (save memory)")
        self.act_ckpt_check.setChecked(True)
        self.act_ckpt_check.setStyleSheet(CHECK_STYLE)
        lay.addWidget(self.act_ckpt_check)

        return g

    def _build_overrides_group(self):
        g = QGroupBox("Config Overrides (optional)")
        lay = QVBoxLayout(g)

        hint = QLabel(
            "Add key=value overrides, one per line. These are passed directly to "
            "the recipe as OmegaConf-style overrides. Examples:\n"
            "  epochs=3\n"
            "  optimizer.lr=2e-5\n"
            "  batch_size=4\n"
            "  dataset.source=tatsu-lab/alpaca")
        hint.setStyleSheet(LABEL_SECONDARY)
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self.overrides_edit = QPlainTextEdit()
        self.overrides_edit.setMaximumHeight(100)
        self.overrides_edit.setPlaceholderText("epochs=3\noptimizer.lr=2e-5\nbatch_size=4")
        self.overrides_edit.setStyleSheet(
            "QPlainTextEdit { font-family: 'Consolas', monospace; font-size: 12px; "
            "background: #ffffff; border: 1px solid #c9cdd3; border-radius: 4px; "
            "padding: 8px; color: #1a1a2e; }")
        lay.addWidget(self.overrides_edit)

        return g

    def _build_summary_card(self):
        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet(
            "QFrame{background:#e8edf3;"
            "border:1px solid #c9cdd3;border-radius:4px;padding:8px 14px;}")
        lay = QHBoxLayout(self.summary_frame)
        lay.setContentsMargins(10, 6, 10, 6)
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("font-size:13px;color:#394867;")
        self.summary_label.setTextFormat(Qt.RichText)
        self.summary_label.setWordWrap(True)
        lay.addWidget(self.summary_label)
        return self.summary_frame

    def _build_buttons(self):
        lay = QHBoxLayout()
        self.start_btn = QPushButton("START FULL FINE-TUNING")
        self.start_btn.setMinimumHeight(48)
        self.start_btn.setStyleSheet(
            BTN_PRIMARY_LARGE
            .replace("#394867", "#3a7d44")
            .replace("#212d40", "#2d6235")
            .replace("#14213d", "#1f5029"))
        self.start_btn.clicked.connect(self.start_training)
        lay.addWidget(self.start_btn)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setMinimumHeight(48)
        self.stop_btn.setStyleSheet(
            BTN_PRIMARY_LARGE
            .replace("#394867", "#c0392b")
            .replace("#212d40", "#96281b")
            .replace("#14213d", "#7a1512"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        lay.addWidget(self.stop_btn)
        return lay

    def _build_progress(self):
        lay = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)
        lay.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(200)
        lay.addWidget(self.status_label)
        return lay

    def _build_log(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(5, 5, 5, 5)
        hdr = QHBoxLayout()
        t = QLabel("Training Log")
        t.setStyleSheet(LABEL_HEADING)
        hdr.addWidget(t)
        cb = QPushButton("Clear")
        cb.setFixedWidth(80)
        cb.setStyleSheet(BTN_SECONDARY)
        cb.clicked.connect(lambda: self.log_edit.clear())
        hdr.addWidget(cb)
        self.clear_on_start_check = QCheckBox("Clear on start")
        self.clear_on_start_check.setChecked(True)
        self.clear_on_start_check.setStyleSheet(LABEL_SECONDARY)
        hdr.addWidget(self.clear_on_start_check)
        hdr.addStretch()
        lay.addLayout(hdr)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("""
            QPlainTextEdit { background-color:#1a1a2e; color:#a3d977;
                font-family:'Consolas',monospace; font-size:12px;
                border-radius:4px; padding:10px; }
        """)
        self.log_edit.setMaximumBlockCount(2000)
        lay.addWidget(self.log_edit)
        return w

    def _on_recipe_changed(self, _index=None):
        recipe_name = self.recipe_combo.currentData()
        self.config_combo.clear()
        if recipe_name:
            configs = list_configs_for_recipe(recipe_name)
            for c in configs:
                self.config_combo.addItem(c["name"], c["file_path"])
        self._update_summary()

    def _update_gpu_info(self):
        idx = self.device_combo.currentIndex()
        gpus = _gpu_info()
        if idx <= 0 or not gpus:
            self.gpu_info_frame.setVisible(False)
            return
        gi = gpus[min(idx - 1, len(gpus) - 1)]
        self.gpu_info_frame.setVisible(True)
        self.gpu_name_label.setText(gi["name"])
        self.gpu_mem_label.setText(
            f"VRAM: {gi['total_mb']:.0f} MB total  |  ~{gi['free_mb']:.0f} MB free")

    def _update_summary(self):
        if not hasattr(self, 'summary_label'):
            return
        recipe = self.recipe_combo.currentData() or "N/A"
        config = self.config_combo.currentText() or "N/A"
        nproc = self.nproc_spin.value() if hasattr(self, 'nproc_spin') else 1
        self.summary_label.setText(
            f"<b style='color:#1f5029;'>Full Fine-tuning</b>"
            f" &nbsp;|&nbsp; <b>Recipe:</b> {recipe}"
            f" &nbsp;|&nbsp; <b>Config:</b> {config}"
            f" &nbsp;|&nbsp; <b>GPUs:</b> {nproc}")

    def _browse_ckpt_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Checkpoint Directory", self.project_root)
        if path:
            self.ckpt_dir_edit.setText(path)

    def _browse_dir(self, line_edit):
        path = QFileDialog.getExistingDirectory(
            self, "Select Directory", line_edit.text() or self.project_root)
        if path:
            line_edit.setText(path)

    def start_training(self):
        recipe_name = self.recipe_combo.currentData()
        config_name = self.config_combo.currentText()

        if not recipe_name or not config_name:
            QMessageBox.warning(self, "Error",
                "Please select a recipe and config first.")
            return

        output_dir = self.output_dir_edit.text().strip()
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(self.project_root, output_dir)
        os.makedirs(output_dir, exist_ok=True)

        nproc = self.nproc_spin.value()
        is_distributed = "distributed" in recipe_name and nproc > 1

        if is_distributed:
            cmd = [
                sys.executable, "-m", "torchtune._cli.tune",
                "run",
                "--nproc_per_node", str(nproc),
                recipe_name,
                "--config", config_name,
            ]
        else:
            cmd = [
                sys.executable, "-m", "torchtune._cli.tune",
                "run",
                recipe_name,
                "--config", config_name,
            ]

        cmd.append(f"output_dir={output_dir}")

        ckpt_dir = self.ckpt_dir_edit.text().strip()
        if ckpt_dir:
            cmd.append(f"checkpointer.checkpoint_dir={ckpt_dir}")

        if self.act_ckpt_check.isChecked():
            cmd.append("enable_activation_checkpointing=True")

        overrides = self.overrides_edit.toPlainText().strip()
        if overrides:
            for line in overrides.splitlines():
                line = line.strip()
                if line and "=" in line:
                    cmd.append(line)

        if self.clear_on_start_check.isChecked():
            self.log_edit.clear()

        self.log_edit.appendPlainText("=" * 60)
        self.log_edit.appendPlainText(f"  Full Fine-tuning")
        self.log_edit.appendPlainText(f"  Recipe : {recipe_name}")
        self.log_edit.appendPlainText(f"  Config : {config_name}")
        self.log_edit.appendPlainText(f"  GPUs   : {nproc}")
        self.log_edit.appendPlainText(f"  Output : {output_dir}")
        self.log_edit.appendPlainText("=" * 60)
        self.log_edit.appendPlainText(f"Command: {' '.join(cmd)}")
        self.log_edit.appendPlainText("=" * 60)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Training...")
        self.progress_bar.setRange(0, 0)

        self.worker = TrainingWorker(cmd, self.project_root)
        self.worker.log_signal.connect(self.on_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        self.training_started.emit()

    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self.log_edit.appendPlainText("\nTraining stopped by user")

    def on_log(self, text):
        self.log_edit.appendPlainText(text)
        if "Step" in text or "step" in text:
            try:
                import re
                m = re.search(r'[Ss]tep\s+(\d+)', text)
                if m:
                    step = int(m.group(1))
                    self.progress_bar.setRange(0, 100)
                    self.status_label.setText(f"Step {step}")
            except Exception:
                pass

    def on_finished(self, code):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        if code == 0:
            self.progress_bar.setValue(100)
            self.status_label.setText("Complete!")
            self.log_edit.appendPlainText("\nTraining completed!")
            QMessageBox.information(self, "Success", "Full fine-tuning completed!")
        else:
            self.progress_bar.setValue(0)
            self.status_label.setText("Failed")
            self.log_edit.appendPlainText(f"\nTraining failed (code: {code})")
        self.training_stopped.emit()
