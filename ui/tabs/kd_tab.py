"""
Knowledge Distillation Tab - Run torchtune KD recipes via the UI.

Train a smaller student model by distilling knowledge from a larger teacher.
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

import torch

from ui.styles import (
    BTN_DANGER, BTN_PRIMARY_LARGE, BTN_SECONDARY,
    CHECK_STYLE, COMBO_STYLE, EDIT_STYLE, LABEL_HEADING,
    LABEL_SECONDARY, SPIN_STYLE,
)
from ui.helpers import get_project_root, list_recipes, list_configs_for_recipe


class KDWorker(QThread):
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


class KDTab(QWidget):
    """Knowledge Distillation via torchtune."""

    training_started = pyqtSignal()
    training_stopped = pyqtSignal()

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
        cl.setSpacing(4)

        cl.addWidget(self._build_info_banner())

        row1 = QHBoxLayout()
        row1.addWidget(self._build_recipe_group(), 5)
        row1.addWidget(self._build_device_group(), 3)
        cl.addLayout(row1)

        cl.addWidget(self._build_paths_group())
        cl.addWidget(self._build_overrides_group())
        cl.addWidget(self._build_summary_card())
        cl.addLayout(self._build_buttons())
        cl.addLayout(self._build_progress())

        scroll.setWidget(config_w)
        splitter.addWidget(scroll)
        splitter.addWidget(self._build_log())
        splitter.setSizes([460, 280])
        main_layout.addWidget(splitter)

        self._update_summary()

    def _build_info_banner(self):
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame{background:#fef9e7;border:1px solid #f9e79f;"
            "border-radius:6px;padding:10px 14px;}")
        lay = QVBoxLayout(frame)
        lay.setSpacing(4)
        title = QLabel("Knowledge Distillation Training")
        title.setStyleSheet("font-size:14px;font-weight:700;color:#7d6608;")
        lay.addWidget(title)
        desc = QLabel(
            "Train a smaller student model by distilling knowledge from a larger "
            "teacher model. The student learns to mimic the teacher's outputs.\n\n"
            "Torchtune supports single-device and distributed KD recipes. "
            "Select a recipe and config, set paths to teacher/student checkpoints, "
            "and add any config overrides.")
        desc.setStyleSheet("font-size:12px;color:#5d4e37;")
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
        self.recipe_combo.setMinimumWidth(350)
        for r in self._all_recipes:
            if "knowledge_distillation" in r["name"]:
                label = r["name"]
                if r["supports_distributed"]:
                    label += "  (multi-GPU)"
                self.recipe_combo.addItem(label, r["name"])
        if self.recipe_combo.count() == 0:
            self.recipe_combo.addItem("knowledge_distillation_single_device",
                                       "knowledge_distillation_single_device")
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
        return g

    def _build_paths_group(self):
        g = QGroupBox("Model Paths")
        lay = QGridLayout(g)

        lay.addWidget(QLabel("Teacher Checkpoint Dir:"), 0, 0)
        self.teacher_dir_edit = QLineEdit()
        self.teacher_dir_edit.setPlaceholderText("Path to teacher model checkpoints")
        self.teacher_dir_edit.setStyleSheet(EDIT_STYLE)
        lay.addWidget(self.teacher_dir_edit, 0, 1)
        btn1 = QPushButton("Browse")
        btn1.setFixedWidth(80)
        btn1.setStyleSheet(BTN_SECONDARY)
        btn1.clicked.connect(lambda: self._browse_dir(self.teacher_dir_edit))
        lay.addWidget(btn1, 0, 2)

        lay.addWidget(QLabel("Student Checkpoint Dir:"), 1, 0)
        self.student_dir_edit = QLineEdit()
        self.student_dir_edit.setPlaceholderText("Path to student model checkpoints")
        self.student_dir_edit.setStyleSheet(EDIT_STYLE)
        lay.addWidget(self.student_dir_edit, 1, 1)
        btn2 = QPushButton("Browse")
        btn2.setFixedWidth(80)
        btn2.setStyleSheet(BTN_SECONDARY)
        btn2.clicked.connect(lambda: self._browse_dir(self.student_dir_edit))
        lay.addWidget(btn2, 1, 2)

        lay.addWidget(QLabel("Output Dir:"), 2, 0)
        self.output_dir_edit = QLineEdit("output/kd_training")
        self.output_dir_edit.setStyleSheet(EDIT_STYLE)
        lay.addWidget(self.output_dir_edit, 2, 1)
        btn3 = QPushButton("Browse")
        btn3.setFixedWidth(80)
        btn3.setStyleSheet(BTN_SECONDARY)
        btn3.clicked.connect(lambda: self._browse_dir(self.output_dir_edit))
        lay.addWidget(btn3, 2, 2)

        return g

    def _build_device_group(self):
        g = QGroupBox("Device")
        lay = QVBoxLayout(g)

        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                self.device_combo.addItem(f"GPU {i}: {name[:25]}")
            self.device_combo.setCurrentIndex(1)
        self.device_combo.setStyleSheet(COMBO_STYLE)
        lay.addWidget(self.device_combo)

        nproc_lay = QHBoxLayout()
        nproc_lay.addWidget(QLabel("Num GPUs:"))
        self.nproc_spin = QSpinBox()
        self.nproc_spin.setRange(1, 8)
        self.nproc_spin.setValue(1)
        self.nproc_spin.setStyleSheet(SPIN_STYLE)
        nproc_lay.addWidget(self.nproc_spin)
        nproc_lay.addStretch()
        lay.addLayout(nproc_lay)

        self.act_ckpt_check = QCheckBox("Activation Checkpointing")
        self.act_ckpt_check.setChecked(True)
        self.act_ckpt_check.setStyleSheet(CHECK_STYLE)
        lay.addWidget(self.act_ckpt_check)

        return g

    def _build_overrides_group(self):
        g = QGroupBox("Config Overrides (optional)")
        lay = QVBoxLayout(g)
        hint = QLabel(
            "Override config values. Examples: epochs=5, kd_loss_weight=0.5, "
            "optimizer.lr=1e-5")
        hint.setStyleSheet(LABEL_SECONDARY)
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.overrides_edit = QPlainTextEdit()
        self.overrides_edit.setMaximumHeight(80)
        self.overrides_edit.setPlaceholderText("epochs=5\nkd_loss_weight=0.5")
        self.overrides_edit.setStyleSheet(
            "QPlainTextEdit { font-family: 'Consolas', monospace; font-size: 12px; "
            "background: #ffffff; border: 1px solid #c9cdd3; border-radius: 4px; "
            "padding: 8px; color: #1a1a2e; }")
        lay.addWidget(self.overrides_edit)
        return g

    def _build_summary_card(self):
        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet(
            "QFrame{background:#fef9e7;"
            "border:1px solid #f9e79f;border-radius:4px;padding:8px 14px;}")
        lay = QHBoxLayout(self.summary_frame)
        lay.setContentsMargins(10, 6, 10, 6)
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("font-size:13px;color:#7d6608;")
        self.summary_label.setTextFormat(Qt.RichText)
        self.summary_label.setWordWrap(True)
        lay.addWidget(self.summary_label)
        return self.summary_frame

    def _build_buttons(self):
        lay = QHBoxLayout()
        self.start_btn = QPushButton("START DISTILLATION")
        self.start_btn.setMinimumHeight(48)
        self.start_btn.setStyleSheet(
            BTN_PRIMARY_LARGE
            .replace("#394867", "#7d6608")
            .replace("#212d40", "#5d4e37")
            .replace("#14213d", "#3d3225"))
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
        t = QLabel("Knowledge Distillation Log")
        t.setStyleSheet(LABEL_HEADING)
        hdr.addWidget(t)
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
            QPlainTextEdit { background-color:#1a1a2e; color:#f9e79f;
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

    def _update_summary(self):
        if not hasattr(self, 'summary_label'):
            return
        recipe = self.recipe_combo.currentData() or "N/A"
        config = self.config_combo.currentText() or "N/A"
        self.summary_label.setText(
            f"<b style='color:#7d6608;'>Knowledge Distillation</b>"
            f" &nbsp;|&nbsp; <b>Recipe:</b> {recipe}"
            f" &nbsp;|&nbsp; <b>Config:</b> {config}")

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
                "run", "--nproc_per_node", str(nproc),
                recipe_name, "--config", config_name,
            ]
        else:
            cmd = [
                sys.executable, "-m", "torchtune._cli.tune",
                "run", recipe_name, "--config", config_name,
            ]

        cmd.append(f"output_dir={output_dir}")

        teacher_dir = self.teacher_dir_edit.text().strip()
        if teacher_dir:
            cmd.append(f"teacher_checkpointer.checkpoint_dir={teacher_dir}")

        student_dir = self.student_dir_edit.text().strip()
        if student_dir:
            cmd.append(f"checkpointer.checkpoint_dir={student_dir}")

        if self.act_ckpt_check.isChecked():
            cmd.append("enable_activation_checkpointing=True")

        overrides = self.overrides_edit.toPlainText().strip()
        if overrides:
            for line in overrides.splitlines():
                line = line.strip()
                if line and "=" in line:
                    cmd.append(line)

        self.log_edit.clear()
        self.log_edit.appendPlainText("=" * 60)
        self.log_edit.appendPlainText("  Knowledge Distillation Training")
        self.log_edit.appendPlainText(f"  Recipe : {recipe_name}")
        self.log_edit.appendPlainText(f"  Config : {config_name}")
        self.log_edit.appendPlainText(f"  Output : {output_dir}")
        self.log_edit.appendPlainText("=" * 60)
        self.log_edit.appendPlainText(f"Command: {' '.join(cmd)}")
        self.log_edit.appendPlainText("=" * 60 + "\n")

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Training...")
        self.progress_bar.setRange(0, 0)

        self.worker = KDWorker(cmd, self.project_root)
        self.worker.log_signal.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self.training_started.emit()

    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self.log_edit.appendPlainText("\nTraining stopped by user")

    def _on_log(self, text):
        self.log_edit.appendPlainText(text)
        if "Step" in text or "step" in text:
            try:
                import re
                m = re.search(r'[Ss]tep\s+(\d+)', text)
                if m:
                    self.progress_bar.setRange(0, 100)
                    self.status_label.setText(f"Step {m.group(1)}")
            except Exception:
                pass

    def _on_finished(self, code):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        if code == 0:
            self.progress_bar.setValue(100)
            self.status_label.setText("Complete!")
            self.log_edit.appendPlainText("\nKnowledge Distillation completed!")
            QMessageBox.information(self, "Success",
                "Knowledge Distillation completed!\n\n"
                "Student model saved to the output directory.")
        else:
            self.progress_bar.setValue(0)
            self.status_label.setText("Failed")
            self.log_edit.appendPlainText(f"\nTraining failed (code: {code})")
        self.training_stopped.emit()
