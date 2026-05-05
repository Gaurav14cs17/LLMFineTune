"""
Quantization Tab - Run torchtune's quantize recipe to compress models.

Supports INT8, INT4, and other quantization schemes via torchao.
"""

import os
import sys
import subprocess
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QComboBox, QSpinBox, QCheckBox,
    QFileDialog, QProgressBar, QMessageBox,
    QGridLayout, QFrame, QScrollArea, QPlainTextEdit, QSplitter,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from ui.styles import (
    BTN_PRIMARY_LARGE, BTN_SECONDARY,
    COMBO_STYLE, EDIT_STYLE, LABEL_HEADING, LABEL_SECONDARY,
    SPIN_STYLE, CHECK_STYLE,
)
from ui.helpers import get_project_root, list_recipes, list_configs_for_recipe


class QuantWorker(QThread):
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
                self.process.wait(timeout=120)
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
                self.process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self.process.kill()
                except OSError:
                    pass


class QuantizationTab(QWidget):
    """Quantize LLMs using torchtune's quantize recipe."""

    def __init__(self):
        super().__init__()
        self.worker = None
        self.project_root = get_project_root()
        self._all_recipes = list_recipes()
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Vertical)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        config_w = QWidget()
        cl = QVBoxLayout(config_w)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(8)

        cl.addWidget(self._build_info_banner())

        row1 = QHBoxLayout()
        row1.addWidget(self._build_recipe_group(), 5)
        row1.addWidget(self._build_quant_options_group(), 4)
        cl.addLayout(row1)

        cl.addWidget(self._build_paths_group())
        cl.addWidget(self._build_overrides_group())
        cl.addLayout(self._build_buttons())

        scroll.setWidget(config_w)
        splitter.addWidget(scroll)
        splitter.addWidget(self._build_log())
        splitter.setSizes([380, 320])
        main_layout.addWidget(splitter)

    def _build_info_banner(self):
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame{background:#e8f5e9;border:1px solid #a5d6a7;"
            "border-radius:6px;padding:10px 14px;}")
        lay = QVBoxLayout(frame)
        lay.setSpacing(4)
        title = QLabel("Model Quantization")
        title.setStyleSheet("font-size:14px;font-weight:700;color:#2e7d32;")
        lay.addWidget(title)
        desc = QLabel(
            "Quantize fine-tuned models to reduce size and improve inference speed. "
            "Uses torchtune's quantize recipe powered by torchao.\n\n"
            "Supported: INT8 (dynamic/weight-only), INT4 (GPTQ-style), "
            "QAT (quantization-aware training). Select a recipe + config, "
            "specify paths, and click Quantize.")
        desc.setStyleSheet("font-size:12px;color:#1b5e20;")
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
        self.recipe_combo.setMinimumWidth(300)
        for r in self._all_recipes:
            if "quantiz" in r["name"].lower() or "qat" in r["name"].lower():
                label = r["name"]
                if r["supports_distributed"]:
                    label += "  (multi-GPU)"
                self.recipe_combo.addItem(label, r["name"])
        if self.recipe_combo.count() == 0:
            self.recipe_combo.addItem("quantize", "quantize")
        self.recipe_combo.currentIndexChanged.connect(self._on_recipe_changed)
        recipe_lay.addWidget(self.recipe_combo)
        recipe_lay.addStretch()
        lay.addLayout(recipe_lay)

        config_lay = QHBoxLayout()
        config_lay.addWidget(QLabel("Config:"))
        self.config_combo = QComboBox()
        self.config_combo.setStyleSheet(COMBO_STYLE)
        self.config_combo.setMinimumWidth(350)
        config_lay.addWidget(self.config_combo)
        config_lay.addStretch()
        lay.addLayout(config_lay)

        self._on_recipe_changed()
        return g

    def _build_quant_options_group(self):
        g = QGroupBox("Quantization Options")
        lay = QVBoxLayout(g)

        method_lay = QHBoxLayout()
        method_lay.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "INT8 Dynamic Quantization",
            "INT8 Weight-Only",
            "INT4 Weight-Only (GPTQ-style)",
            "INT4 + Groupwise",
        ])
        self.method_combo.setStyleSheet(COMBO_STYLE)
        self.method_combo.setToolTip(
            "INT8 Dynamic: Fastest to apply, good balance\n"
            "INT8 Weight-Only: Quantize only weights\n"
            "INT4: Maximum compression, some quality loss\n"
            "INT4 + Groupwise: Better quality than plain INT4")
        method_lay.addWidget(self.method_combo)
        lay.addLayout(method_lay)

        self.group_size_lay = QHBoxLayout()
        self.group_size_lay.addWidget(QLabel("Group Size:"))
        self.group_size_spin = QSpinBox()
        self.group_size_spin.setRange(32, 256)
        self.group_size_spin.setValue(128)
        self.group_size_spin.setStyleSheet(SPIN_STYLE)
        self.group_size_spin.setToolTip("Group size for groupwise quantization (lower = better quality)")
        self.group_size_lay.addWidget(self.group_size_spin)
        self.group_size_lay.addStretch()
        lay.addLayout(self.group_size_lay)

        return g

    def _build_paths_group(self):
        g = QGroupBox("Paths")
        lay = QGridLayout(g)

        lay.addWidget(QLabel("Checkpoint Dir:"), 0, 0)
        self.ckpt_dir_edit = QLineEdit()
        self.ckpt_dir_edit.setPlaceholderText("Path to model checkpoint directory")
        self.ckpt_dir_edit.setStyleSheet(EDIT_STYLE)
        lay.addWidget(self.ckpt_dir_edit, 0, 1)
        btn1 = QPushButton("Browse")
        btn1.setFixedWidth(80)
        btn1.setStyleSheet(BTN_SECONDARY)
        btn1.clicked.connect(lambda: self._browse_dir(self.ckpt_dir_edit))
        lay.addWidget(btn1, 0, 2)

        lay.addWidget(QLabel("Output Dir:"), 1, 0)
        self.output_dir_edit = QLineEdit("output/quantized")
        self.output_dir_edit.setStyleSheet(EDIT_STYLE)
        lay.addWidget(self.output_dir_edit, 1, 1)
        btn2 = QPushButton("Browse")
        btn2.setFixedWidth(80)
        btn2.setStyleSheet(BTN_SECONDARY)
        btn2.clicked.connect(lambda: self._browse_dir(self.output_dir_edit))
        lay.addWidget(btn2, 1, 2)

        return g

    def _build_overrides_group(self):
        g = QGroupBox("Config Overrides (optional)")
        lay = QVBoxLayout(g)
        self.overrides_edit = QPlainTextEdit()
        self.overrides_edit.setMaximumHeight(60)
        self.overrides_edit.setPlaceholderText("quantizer.groupsize=64")
        self.overrides_edit.setStyleSheet(
            "QPlainTextEdit { font-family: 'Consolas', monospace; font-size: 12px; "
            "background: #ffffff; border: 1px solid #c9cdd3; border-radius: 4px; "
            "padding: 8px; color: #1a1a2e; }")
        lay.addWidget(self.overrides_edit)
        return g

    def _build_buttons(self):
        lay = QHBoxLayout()

        self.quant_btn = QPushButton("QUANTIZE MODEL")
        self.quant_btn.setMinimumHeight(48)
        self.quant_btn.setStyleSheet(
            BTN_PRIMARY_LARGE
            .replace("#394867", "#2e7d32")
            .replace("#212d40", "#1b5e20")
            .replace("#14213d", "#0d3b13"))
        self.quant_btn.clicked.connect(self.start_quantization)
        lay.addWidget(self.quant_btn)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setMinimumHeight(48)
        self.stop_btn.setStyleSheet(
            BTN_PRIMARY_LARGE
            .replace("#394867", "#c0392b")
            .replace("#212d40", "#96281b")
            .replace("#14213d", "#7a1512"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_quantization)
        lay.addWidget(self.stop_btn)

        return lay

    def _build_log(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(5, 5, 5, 5)

        hdr = QHBoxLayout()
        t = QLabel("Quantization Log")
        t.setStyleSheet(LABEL_HEADING)
        hdr.addWidget(t)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(LABEL_SECONDARY)
        hdr.addWidget(self.status_label)

        cb = QPushButton("Clear")
        cb.setFixedWidth(80)
        cb.setStyleSheet(BTN_SECONDARY)
        cb.clicked.connect(lambda: self.log_edit.clear())
        hdr.addWidget(cb)
        hdr.addStretch()
        lay.addLayout(hdr)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("""
            QPlainTextEdit { background-color:#1a1a2e; color:#a5d6a7;
                font-family:'Consolas',monospace; font-size:12px;
                border-radius:4px; padding:10px; }
        """)
        self.log_edit.setMaximumBlockCount(1000)
        lay.addWidget(self.log_edit)
        return w

    def _on_recipe_changed(self, _index=None):
        recipe_name = self.recipe_combo.currentData()
        self.config_combo.clear()
        if recipe_name:
            configs = list_configs_for_recipe(recipe_name)
            for c in configs:
                self.config_combo.addItem(c["name"], c["file_path"])

    def _browse_dir(self, line_edit):
        path = QFileDialog.getExistingDirectory(
            self, "Select Directory", line_edit.text() or self.project_root)
        if path:
            line_edit.setText(path)

    def start_quantization(self):
        recipe_name = self.recipe_combo.currentData() or "quantize"
        config_name = self.config_combo.currentText()

        ckpt_dir = self.ckpt_dir_edit.text().strip()
        if not ckpt_dir:
            QMessageBox.warning(self, "Error",
                "Please specify the checkpoint directory.")
            return

        output_dir = self.output_dir_edit.text().strip()
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(self.project_root, output_dir)
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            sys.executable, "-m", "torchtune._cli.tune",
            "run", recipe_name,
        ]

        if config_name:
            cmd.extend(["--config", config_name])

        cmd.append(f"output_dir={output_dir}")
        cmd.append(f"checkpointer.checkpoint_dir={ckpt_dir}")

        method = self.method_combo.currentText()
        if "INT8 Dynamic" in method:
            cmd.append("quantizer._component_=torchtune.training.quantization.Int8DynActInt4WeightQuantizer")
        elif "INT8 Weight" in method:
            cmd.append("quantizer._component_=torchtune.training.quantization.Int8WeightOnlyQuantizer")
        elif "INT4" in method:
            cmd.append("quantizer._component_=torchtune.training.quantization.Int4WeightOnlyQuantizer")
            cmd.append(f"quantizer.groupsize={self.group_size_spin.value()}")

        overrides = self.overrides_edit.toPlainText().strip()
        if overrides:
            for line in overrides.splitlines():
                line = line.strip()
                if line and "=" in line:
                    cmd.append(line)

        self.log_edit.clear()
        self.log_edit.appendPlainText("=" * 60)
        self.log_edit.appendPlainText("  Model Quantization")
        self.log_edit.appendPlainText(f"  Recipe : {recipe_name}")
        self.log_edit.appendPlainText(f"  Method : {method}")
        self.log_edit.appendPlainText(f"  Source : {ckpt_dir}")
        self.log_edit.appendPlainText(f"  Output : {output_dir}")
        self.log_edit.appendPlainText("=" * 60)
        self.log_edit.appendPlainText(f"Command: {' '.join(cmd)}")
        self.log_edit.appendPlainText("=" * 60 + "\n")

        self.quant_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Quantizing...")

        self.worker = QuantWorker(cmd, self.project_root)
        self.worker.log_signal.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def stop_quantization(self):
        if self.worker:
            self.worker.stop()

    def _on_log(self, text):
        self.log_edit.appendPlainText(text)

    def _on_finished(self, code):
        self.quant_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if code == 0:
            self.status_label.setText("Quantization complete!")
            self.log_edit.appendPlainText("\nQuantization completed!")
            QMessageBox.information(self, "Success",
                "Model quantized successfully!")
        else:
            self.status_label.setText("Quantization failed")
            self.log_edit.appendPlainText(f"\nQuantization failed (code: {code})")
