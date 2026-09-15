# -*- coding: utf-8 -*-
"""
PDF 分析报告生成（P2.7）。
只用 reportlab（含内置 CID 中文字体 STSong-Light，无需外部字体文件，
兼容 Streamlit Cloud 的 Linux 容器），图表用 reportlab.graphics 原生绘制。
"""
from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pandas as pd

FONT = "STSong-Light"
DISCLAIMER = (
    "免责声明：本报告由个人学习项目自动生成，仅为基于公开历史数据的量化模拟测算，"
    "不构成个性化投资建议或任何证券买卖要约。历史表现不代表未来收益，市场有风险，"
    "投资需谨慎。任何机构或个人使用本报告内容进行投资决策，风险与后果由使用者自行承担。"
)


def _register_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(FONT))
    except Exception:
        pass


def _styles():
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    return {
        "title": ParagraphStyle("t", fontName=FONT, fontSize=20, leading=28,
                                spaceAfter=6),
        "h2": ParagraphStyle("h2", fontName=FONT, fontSize=13, leading=18,
                             spaceBefore=14, spaceAfter=6,
                             textColor=colors.HexColor("#1F3864")),
        "body": ParagraphStyle("b", fontName=FONT, fontSize=10, leading=16),
        "small": ParagraphStyle("s", fontName=FONT, fontSize=8.5, leading=13,
                                textColor=colors.HexColor("#666666")),
    }


def _table(data, col_widths=None, header=True):
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    t = Table(data, colWidths=col_widths)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F5F7FB")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(style))
    return t


def _nav_drawing(nav: pd.Series, benchmark: tuple | None = None):
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib import colors
    ys = nav.values.astype(float)
    n = len(ys)
    step = max(1, n // 250)                    # 抽样，避免 PDF 过大
    xs = np.arange(0, n, step)
    data = [list(zip(xs, ys[::step]))]
    if benchmark is not None:
        bnav = benchmark[1].reindex(nav.index).ffill().dropna()
        b = bnav.values.astype(float)
        m = min(len(b), n)
        data.append(list(zip(np.arange(0, m, step), b[:m:step])))
    d = Drawing(460, 180)
    lp = LinePlot()
    lp.x = 30
    lp.y = 25
    lp.width = 420
    lp.height = 140
    lp.data = data
    lp.lines[0].strokeColor = colors.HexColor("#4472C4")
    lp.lines[0].strokeWidth = 1.5
    if len(data) > 1:
        lp.lines[1].strokeColor = colors.HexColor("#999999")
        lp.lines[1].strokeWidth = 1
        lp.lines[1].strokeDashArray = [3, 2]
    d.add(lp)
    return d


def _pie_drawing(labels, weights):
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.lib import colors
    palette = ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5",
               "#70AD47", "#264478", "#9E480E", "#636363", "#997300"]
    d = Drawing(460, 170)
    pie = Pie()
    pie.x = 80
    pie.y = 15
    pie.width = 140
    pie.height = 140
    pie.data = [max(float(w), 0.001) for w in weights]
    pie.labels = [f"{l} {w:.0%}" for l, w in zip(labels, weights)]
    for i in range(len(weights)):
        pie.slices[i].fillColor = colors.HexColor(palette[i % len(palette)])
        pie.slices[i].fontName = FONT
        pie.slices[i].fontSize = 7.5
    d.add(pie)
    return d


def _bar_drawing(labels, money_pct, risk_pct):
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib import colors
    d = Drawing(460, 190)
    bc = VerticalBarChart()
    bc.x = 35
    bc.y = 45
    bc.width = 410
    bc.height = 130
    bc.data = [list(money_pct), list(risk_pct)]
    bc.bars[0].fillColor = colors.HexColor("#9DB2D5")
    bc.bars[1].fillColor = colors.HexColor("#C00000")
    bc.categoryAxis.categoryNames = list(labels)
    bc.categoryAxis.labels.fontName = FONT
    bc.categoryAxis.labels.fontSize = 7
    bc.categoryAxis.labels.angle = 20
    bc.valueAxis.valueMin = 0
    d.add(bc)
    return d


def build_portfolio_pdf(
    title: str,
    subtitle: str,
    metrics: dict,                 # {指标名: 值字符串} 四个核心指标
    weight_rows: list[list],       # [[资产, 产品, 比例, 金额], ...]
    nav: pd.Series,
    benchmark=None,                # (名称, Series) 或 None
    rc_df: pd.DataFrame | None = None,   # 风险归因表（含 资产/资金占比/风险贡献占比）
    analysis_text: str = "",
    extra_notes: list[str] | None = None,
) -> bytes:
    """生成完整 PDF 报告，返回 bytes（供 st.download_button 或写文件）。"""
    _register_font()
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=54, rightMargin=54,
                            topMargin=48, bottomMargin=48,
                            title=title, author="个人财富管理配置器")
    story = [
        Paragraph(title, st["title"]),
        Paragraph(f"{subtitle}　·　生成时间：{dt.datetime.now():%Y-%m-%d %H:%M}",
                  st["small"]),
        Spacer(1, 8),
        Paragraph("一、组合核心指标", st["h2"]),
        _table([list(metrics.keys()), list(metrics.values())], header=False),
        Paragraph("二、配置方案", st["h2"]),
        _table([["资产", "具体产品", "配置比例", "配置金额"]] + weight_rows),
        Spacer(1, 6),
        _pie_drawing([r[0] for r in weight_rows],
                     [float(str(r[2]).rstrip("%")) / 100 for r in weight_rows]),
        Paragraph("三、历史净值走势（近5年，起点=1）", st["h2"]),
        _nav_drawing(nav, benchmark),
    ]
    if benchmark is not None:
        story.append(Paragraph(f"灰虚线 = 基准 {benchmark[0]}", st["small"]))
    if rc_df is not None and not rc_df.empty:
        from reportlab.platypus import KeepTogether
        story.append(KeepTogether([
            Paragraph("四、风险归因：资金占比 vs 风险贡献占比", st["h2"]),
            _bar_drawing(rc_df["资产"],
                         rc_df["资金占比"], rc_df["风险贡献占比"]),
            Paragraph("蓝柱 = 资金占比，红柱 = 风险贡献占比。专业配置看风险预算："
                      "某资产资金占比不高但风险贡献很高时，它才是组合波动的主要来源。",
                      st["small"]),
        ]))
    if analysis_text:
        story += [Paragraph("五、自动分析", st["h2"]),
                  Paragraph(analysis_text.replace("**", ""), st["body"])]
    if extra_notes:
        story.append(Paragraph("六、备注", st["h2"]))
        for n in extra_notes:
            story.append(Paragraph("· " + n, st["small"]))
    story += [Spacer(1, 16),
              Paragraph(DISCLAIMER, st["small"])]
    doc.build(story)
    return buf.getvalue()
