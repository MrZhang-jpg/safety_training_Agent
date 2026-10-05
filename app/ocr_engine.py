# -*- coding: utf-8 -*-
"""
本地 OCR 引擎（面向对象）。

使用 rapidocr-onnxruntime：纯 CPU、基于 ONNX，无需安装 Tesseract 等外部二进制，
用于试卷拍照/扫描件的文字识别，输出按阅读顺序排列的文本，供判卷工具解析。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from .config import settings


class OCREngine:
    """RapidOCR 封装。"""

    def __init__(self):
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    @staticmethod
    def _to_ndarray(image: Union[str, Path, "np.ndarray", object]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            from PIL import Image

            with Image.open(str(image)) as im:
                return np.array(im.convert("RGB"))
        if isinstance(image, np.ndarray):
            return image
        # PIL Image
        try:
            return np.array(image.convert("RGB"))
        except Exception:
            pass
        raise TypeError("不支持的图像输入，请传入文件路径、numpy 数组或 PIL Image。")

    def recognize(self, image: Union[str, Path, np.ndarray, object]
                  ) -> list[dict]:
        """识别图像，返回 [{text, score, box}]，按从上到下、从左到右排序。"""
        engine = self._get_engine()
        arr = self._to_ndarray(image)
        result, _ = engine(arr)
        if not result:
            return []

        items = [
            {"box": box, "text": text, "score": float(score)}
            for box, text, score in result
        ]
        # 按行排序：先按 box 顶部 y，再按左侧 x
        def _key(it):
            box = it["box"]
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            return (round(min(ys) / 12), min(xs))

        items.sort(key=_key)
        return items

    def extract_text(self, image: Union[str, Path, np.ndarray, object]) -> str:
        """识别并拼接为纯文本（每行一条）。"""
        items = self.recognize(image)
        return "\n".join(it["text"] for p in [items] for it in p)
