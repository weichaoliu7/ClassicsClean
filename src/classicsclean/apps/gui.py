# -*- coding: utf-8 -*-
"""【第 4 层 · 入口】图形界面：拖拽上传 + 按钮上传，后台线程处理，实时进度。

只调用 process_pdf，不重复任何算法逻辑；处理在子线程、界面刷新在主线程。

设计要点：
  * 拖拽与按钮上传走同一条 `add_files()` 路径，行为完全一致；
  * 处理过程在子线程执行，主线程只负责刷新界面，绝不卡死；
  * 子线程通过 queue 与界面通信，所有 UI 操作都在主线程完成（tkinter 要求）。
"""
from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

from ..config import DEFAULT_PARAMS, VERSION
from ..pipeline.document import process_pdf

try:                                   # 拖拽支持（缺失时降级为仅按钮上传）
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:                      # pragma: no cover - 取决于运行环境
    DND_FILES, TkinterDnD = None, None

BG = "#f5f6f8"
CARD = "#ffffff"
ACCENT = "#2b6cb0"
TEXT = "#1a202c"
MUTED = "#718096"
FONT = "Microsoft YaHei UI"
MONO = "Consolas"

QUALITY_PRESETS = {
    "标准（推荐 · 52 dB）": 52.0,
    "高保真（58 dB · 体积更大）": 58.0,
    "高压缩（46 dB · 体积更小）": 46.0,
}


def _enable_dpi_awareness() -> None:
    """Windows 高分屏下让文字保持清晰。"""
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


class App:
    """主界面。"""

    def __init__(self, root: "tk.Tk") -> None:
        self.root = root
        self.files: List[str] = []
        self.messages: "queue.Queue" = queue.Queue()
        self.running = False
        self.last_outputs: List[str] = []
        self._build()
        self._pump()

    # ---------------- 界面搭建 ----------------
    def _build(self) -> None:
        root = self.root
        root.title(f"ClassicsClean {VERSION} · 古籍扫描件白底清洁工具")
        root.geometry("820x620")
        root.minsize(760, 560)
        root.configure(bg=BG)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD, relief="flat")
        style.configure("TLabel", background=BG, foreground=TEXT, font=(FONT, 10))
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=(FONT, 16, "bold"))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=(FONT, 9))
        style.configure("Accent.TButton", font=(FONT, 10, "bold"))
        style.configure("TProgressbar", troughcolor="#e2e8f0", background=ACCENT)

        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="古籍扫描件白底清洁工具", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="把泛黄、霉斑、透字的扫描 PDF 拉成白底；纸面任意压白，笔画一个灰阶都不动。",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 12))

        # --- 拖拽 / 上传区 ---
        drop = ttk.Frame(outer, style="Card.TFrame", padding=18)
        drop.pack(fill="x")
        drop.columnconfigure(0, weight=1)
        self.drop_label = tk.Label(
            drop,
            text="把 PDF 拖到这里\n或点击下方按钮选择文件",
            bg=CARD,
            fg=ACCENT,
            font=(FONT, 12),
            justify="center",
            padx=20,
            pady=22,
            relief="groove",
            bd=1,
        )
        self.drop_label.grid(row=0, column=0, sticky="ew")
        if TkinterDnD is not None:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self._on_drop)
            self.drop_label.dnd_bind("<<DragEnter>>", self._on_drag_enter)
            self.drop_label.dnd_bind("<<DragLeave>>", self._on_drag_leave)
        else:
            self.drop_label.configure(
                text="点击下方按钮选择文件\n（当前运行环境未加载拖拽组件，拖拽不可用）"
            )

        btns = ttk.Frame(outer)
        btns.pack(fill="x", pady=(10, 6))
        ttk.Button(btns, text="选择 PDF 文件…", command=self._browse).pack(side="left")
        ttk.Button(btns, text="移除选中", command=self._remove_selected).pack(side="left", padx=8)
        ttk.Button(btns, text="清空列表", command=self._clear).pack(side="left")

        # --- 文件列表 ---
        list_card = ttk.Frame(outer, style="Card.TFrame", padding=(10, 8))
        list_card.pack(fill="both", expand=False)
        self.listbox = tk.Listbox(
            list_card, height=4, font=(MONO, 9), relief="flat", bd=0,
            highlightthickness=1, highlightcolor="#cbd5e0", selectbackground=ACCENT,
        )
        self.listbox.pack(fill="x")

        # --- 选项 ---
        opt = ttk.Frame(outer)
        opt.pack(fill="x", pady=(10, 4))
        ttk.Label(opt, text="编码质量：").pack(side="left")
        self.quality = ttk.Combobox(
            opt, values=list(QUALITY_PRESETS), state="readonly", width=24, font=(FONT, 9)
        )
        self.quality.set(list(QUALITY_PRESETS)[0])
        self.quality.pack(side="left")
        self.selfcheck = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="出片后自检（较慢）", variable=self.selfcheck).pack(
            side="left", padx=16
        )

        out_row = ttk.Frame(outer)
        out_row.pack(fill="x", pady=(2, 8))
        ttk.Label(out_row, text="输出目录：").pack(side="left")
        self.outdir = tk.StringVar(value="")
        ttk.Entry(out_row, textvariable=self.outdir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(out_row, text="浏览…", command=self._browse_outdir, width=8).pack(side="left")
        ttk.Label(outer, text="留空 = 输出到源文件同目录，文件名追加「_清洁版」。",
                  style="Muted.TLabel").pack(anchor="w")

        # --- 进度 ---
        self.progress = ttk.Progressbar(outer, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(8, 2))
        self.status = ttk.Label(outer, text="就绪", style="Muted.TLabel")
        self.status.pack(anchor="w")

        self.log = tk.Text(outer, height=9, font=(MONO, 9), relief="flat", bd=1,
                           bg="#ffffff", fg="#2d3748", wrap="none")
        self.log.pack(fill="both", expand=True, pady=(6, 8))
        self.log.configure(state="disabled")

        # --- 底部按钮 ---
        bottom = ttk.Frame(outer)
        bottom.pack(fill="x")
        self.start_btn = ttk.Button(bottom, text="开始处理", style="Accent.TButton",
                                    command=self._start)
        self.start_btn.pack(side="right")
        ttk.Button(bottom, text="打开输出目录", command=self._open_outdir).pack(side="right", padx=8)
        ttk.Button(bottom, text="退出", command=self.root.destroy).pack(side="left")

    # ---------------- 事件 ----------------
    def _on_drag_enter(self, event) -> None:
        self.drop_label.configure(bg="#ebf8ff")
        return event.action

    def _on_drag_leave(self, event) -> None:
        self.drop_label.configure(bg=CARD)
        return event.action

    def _on_drop(self, event) -> None:
        self.drop_label.configure(bg=CARD)
        raw = event.data
        paths = self.root.tk.splitlist(raw) if hasattr(self.root.tk, "splitlist") else raw.split()
        self.add_files([str(p) for p in paths])
        return event.action

    def _browse(self) -> None:
        paths = filedialog.askopenfilenames(
            title="选择 PDF 文件",
            filetypes=[("PDF 文档", "*.pdf"), ("所有文件", "*.*")],
        )
        self.add_files(list(paths))

    def _browse_outdir(self) -> None:
        d = filedialog.askdirectory(title="选择输出目录")
        if d:
            self.outdir.set(d)

    def add_files(self, paths: List[str]) -> None:
        added = 0
        for p in paths:
            p = str(p).strip().strip("{}")
            if not p.lower().endswith(".pdf"):
                continue
            if not os.path.isfile(p):
                continue
            if p in self.files:
                continue
            self.files.append(p)
            self.listbox.insert("end", p)
            added += 1
        if added:
            self._set_status(f"已添加 {added} 个文件，共 {len(self.files)} 个待处理。")
        elif paths:
            messagebox.showwarning("未添加", "只接受真实存在的 .pdf 文件。")

    def _remove_selected(self) -> None:
        for idx in reversed(self.listbox.curselection()):
            self.listbox.delete(idx)
            del self.files[idx]
        self._set_status(f"剩余 {len(self.files)} 个文件。")

    def _clear(self) -> None:
        self.listbox.delete(0, "end")
        self.files.clear()
        self._set_status("已清空。")

    def _open_outdir(self) -> None:
        targets = self.last_outputs or self.files
        if not targets:
            messagebox.showinfo("提示", "还没有可打开的输出。")
            return
        folder = str(Path(targets[0]).parent)
        try:
            os.startfile(folder)          # noqa: S606 - 打开资源管理器，属于预期行为
        except Exception as exc:
            messagebox.showerror("打不开", str(exc))

    def _start(self) -> None:
        if self.running:
            return
        if not self.files:
            messagebox.showinfo("提示", "请先拖入或选择至少一个 PDF 文件。")
            return
        self.running = True
        self.start_btn.configure(state="disabled", text="处理中…")
        self.last_outputs = []
        self.progress["value"] = 0
        self._log_clear()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---------------- 后台处理 ----------------
    def _worker(self) -> None:
        params = DEFAULT_PARAMS.with_db(QUALITY_PRESETS[self.quality.get()])
        outdir = self.outdir.get().strip() or None
        total_files = len(self.files)
        ok = 0
        t_all = time.time()
        for n, src in enumerate(self.files, 1):
            name = Path(src).name
            try:
                self.messages.put(("status", f"正在处理（{n}/{total_files}）：{name}"))
                self.messages.put(("log", f"── 开始：{name}"))
                base = Path(src).stem + "_清洁版.pdf"
                dst = str(Path(outdir) / base) if outdir else str(Path(src).with_name(base))

                def progress(done: int, count: int, text: str) -> None:
                    self.messages.put(("progress", (n - 1) * count + done, count * total_files, text))
                    self.messages.put(("log", text))

                result = process_pdf(src, dst, params, progress=progress,
                                     validate=self.selfcheck.get())
                self.last_outputs.append(result.dst)
                ok += 1
                self.messages.put(
                    ("log",
                     f"── 完成：{result.page_count} 页，"
                     f"{os.path.getsize(src)/1048576:.2f} MB → {result.output_bytes/1048576:.2f} MB，"
                     f"用时 {result.elapsed:.1f}s")
                )
                if result.validation:
                    worst = min(v.psnr for v in result.validation)
                    self.messages.put(
                        ("log", f"   自检：最差 PSNR {worst:.2f} dB，"
                                f"最大像素误差 {max(v.maxerr for v in result.validation)}")
                    )
                self.messages.put(("log", f"   输出：{result.dst}"))
            except Exception as exc:
                self.messages.put(("log", f"── 失败：{name}\n   {exc}"))
        self.messages.put(("done", ok, total_files, time.time() - t_all))

    # ---------------- 主线程刷新 ----------------
    def _pump(self) -> None:
        try:
            while True:
                msg = self.messages.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, done, total, text = msg
                    self.progress["value"] = 100.0 * done / max(1, total)
                    self._set_status(text)
                elif kind == "status":
                    self._set_status(msg[1])
                elif kind == "log":
                    self._log(msg[1])
                elif kind == "done":
                    _, ok, total_files, elapsed = msg
                    self.running = False
                    self.start_btn.configure(state="normal", text="开始处理")
                    self.progress["value"] = 100 if ok else 0
                    self._set_status(f"全部结束：成功 {ok}/{total_files}，总用时 {elapsed:.1f}s")
                    if ok:
                        messagebox.showinfo(
                            "处理完成",
                            f"成功 {ok}/{total_files} 个文件，总用时 {elapsed:.1f} 秒。\n"
                            + "\n".join(self.last_outputs[:5])
                            + ("\n…" if len(self.last_outputs) > 5 else ""),
                        )
                    else:
                        messagebox.showerror("处理失败", "没有文件处理成功，请查看日志。")
        except queue.Empty:
            pass
        self.root.after(120, self._pump)

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _log_clear(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")


def launch() -> int:
    """启动图形界面（阻塞直到窗口关闭）。"""
    _enable_dpi_awareness()
    root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
    App(root)
    root.mainloop()
    return 0
