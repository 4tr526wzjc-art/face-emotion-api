import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def average_of(records, key):
    """计算某项数据的平均值。"""
    values = [
        item[key]
        for item in records
        if key in item
    ]

    if not values:
        return 0.0

    return float(np.mean(values))


def save_csv(records, file_path):
    """保存全部原始检测数据。"""

    if not records:
        return

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=records[0].keys()
        )

        writer.writeheader()
        writer.writerows(records)


def save_chart(records, file_path):
    """生成趋势图 PNG。"""

    if not records:
        return

    times = [item["time_seconds"] for item in records]
    activities = [item["activity"] for item in records]
    attentions = [item["attention"] for item in records]
    tensions = [item["tension"] for item in records]

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS"
    ]

    plt.rcParams["axes.unicode_minus"] = False

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True
    )

    figure.suptitle(
        "数字疗愈过程面部行为趋势图",
        fontsize=16
    )

    axes[0].plot(
        times,
        activities,
        color="#159A9C",
        linewidth=2,
        label="面部活跃度"
    )

    axes[0].plot(
        times,
        tensions,
        color="#E67E22",
        linewidth=1.8,
        label="紧张相关指数"
    )

    axes[0].axhline(
        0.35,
        color="#C0392B",
        linestyle="--",
        linewidth=1,
        label="紧张参考阈值"
    )

    axes[0].set_ylabel("相对指数")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        times,
        attentions,
        color="#3D7EFF",
        linewidth=2,
        label="观看投入度"
    )

    axes[1].axvline(
        60,
        color="gray",
        linestyle=":"
    )

    axes[1].axvline(
        180,
        color="gray",
        linestyle=":"
    )

    axes[1].axvline(
        240,
        color="gray",
        linestyle=":"
    )

    axes[1].set_xlabel("观看时间（秒）")
    axes[1].set_ylabel("相对指数")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    plt.tight_layout()

    plt.savefig(
        file_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


def save_text_report(
    file_path,
    checkpoints,
    records,
    suggestions
):
    """生成中文文字报告。"""

    activity = average_of(records, "activity")
    attention = average_of(records, "attention")
    tension = average_of(records, "tension")

    anxiety_count = sum(
        1 for item in records
        if item.get("anxiety_persistent") == "是"
    )

    anxiety_ratio = anxiety_count / len(records)

    lines = [
        "数字疗愈视频情绪反馈报告",
        "=" * 42,
        "",
        "重要说明：",
        "本报告基于面部关键点、表情变化和头部姿态生成，",
        "仅用于数字疗愈过程中的面部行为趋势反馈。",
        "不构成焦虑症、心理疾病或医学诊断。",
        "",
        "一、整体结果",
        f"平均面部活跃度：{activity:.3f}",
        f"平均观看投入度：{attention:.3f}",
        f"平均紧张相关指数：{tension:.3f}",
        f"持续紧张倾向占比：{anxiety_ratio:.1%}",
        "",
        "二、关键时间节点分析",
    ]

    for second in [60, 180, 240]:
        item = checkpoints.get(second)

        if not item:
            continue

        lines.extend([
            "",
            f"第 {item['minute']} 分钟：{item['state']}",
            f"面部活跃度：{item['activity']:.3f}",
            f"观看投入度：{item['attention']:.3f}",
            f"紧张相关指数：{item['tension']:.3f}",
        ])

    lines.extend([
        "",
        "三、疗愈建议",
    ])

    for index, suggestion in enumerate(
        suggestions,
        start=1
    ):
        lines.append(f"{index}. {suggestion}")

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write("\n".join(lines))


def save_excel(
    file_path,
    checkpoints,
    records,
    suggestions
):
    """生成综合汇总 Excel 文件。"""

    workbook = Workbook()

    title_fill = PatternFill(
        "solid",
        fgColor="159A9C"
    )

    header_fill = PatternFill(
        "solid",
        fgColor="B8E0E0"
    )

    # ======================================================
    # 工作表 1：综合总览
    # ======================================================
    summary_sheet = workbook.active
    summary_sheet.title = "综合总览"

    summary_sheet.merge_cells("A1:F1")

    summary_sheet["A1"] = (
        "数字疗愈视频情绪反馈系统：综合报告"
    )

    summary_sheet["A1"].font = Font(
        size=16,
        bold=True,
        color="FFFFFF"
    )

    summary_sheet["A1"].fill = title_fill

    summary_sheet["A1"].alignment = Alignment(
        horizontal="center",
        vertical="center"
    )

    activity = average_of(records, "activity")
    attention = average_of(records, "attention")
    tension = average_of(records, "tension")

    anxiety_count = sum(
        1 for item in records
        if item.get("anxiety_persistent") == "是"
    )

    anxiety_ratio = anxiety_count / len(records)

    overview_rows = [
        ["平均面部活跃度", round(activity, 3)],
        ["平均观看投入度", round(attention, 3)],
        ["平均紧张相关指数", round(tension, 3)],
        ["持续紧张倾向占比", round(anxiety_ratio, 3)],
        ["有效数据记录数", len(records)],
    ]

    for row, data in enumerate(
        overview_rows,
        start=3
    ):
        summary_sheet.cell(
            row=row,
            column=1,
            value=data[0]
        )

        summary_sheet.cell(
            row=row,
            column=2,
            value=data[1]
        )

    checkpoint_headers = [
        "时间节点",
        "状态趋势",
        "面部活跃度",
        "观看投入度",
        "紧张相关指数",
        "微笑活动"
    ]

    for column, header in enumerate(
        checkpoint_headers,
        start=1
    ):
        cell = summary_sheet.cell(
            row=10,
            column=column,
            value=header
        )

        cell.fill = header_fill
        cell.font = Font(bold=True)
        cell.alignment = Alignment(
            horizontal="center"
        )

    for row, second in enumerate(
        [60, 180, 240],
        start=11
    ):
        item = checkpoints.get(second, {})

        values = [
            f"第 {second // 60} 分钟",
            item.get("state", "数据不足"),
            round(item.get("activity", 0.0), 3),
            round(item.get("attention", 0.0), 3),
            round(item.get("tension", 0.0), 3),
            round(item.get("smile", 0.0), 3),
        ]

        for column, value in enumerate(
            values,
            start=1
        ):
            summary_sheet.cell(
                row=row,
                column=column,
                value=value
            )

    summary_sheet["A16"] = "疗愈建议"
    summary_sheet["A16"].font = Font(bold=True)

    for row, suggestion in enumerate(
        suggestions,
        start=17
    ):
        summary_sheet.merge_cells(
            start_row=row,
            start_column=1,
            end_row=row,
            end_column=6
        )

        summary_sheet.cell(
            row=row,
            column=1,
            value=f"{row - 16}. {suggestion}"
        )

        summary_sheet.cell(
            row=row,
            column=1
        ).alignment = Alignment(
            wrap_text=True,
            vertical="top"
        )

        summary_sheet.row_dimensions[row].height = 42

    for column in range(1, 7):
        summary_sheet.column_dimensions[
            get_column_letter(column)
        ].width = 20

    # ======================================================
    # 工作表 2：全部原始数据
    # ======================================================
    raw_sheet = workbook.create_sheet("全部原始检测数据")

    raw_headers = list(records[0].keys())

    for column, header in enumerate(
        raw_headers,
        start=1
    ):
        cell = raw_sheet.cell(
            row=1,
            column=column,
            value=header
        )

        cell.fill = title_fill
        cell.font = Font(
            bold=True,
            color="FFFFFF"
        )

    for row, record in enumerate(
        records,
        start=2
    ):
        for column, header in enumerate(
            raw_headers,
            start=1
        ):
            raw_sheet.cell(
                row=row,
                column=column,
                value=record[header]
            )

    raw_sheet.freeze_panes = "A2"

    for column in range(1, len(raw_headers) + 1):
        raw_sheet.column_dimensions[
            get_column_letter(column)
        ].width = 16

    # ======================================================
    # 工作表 3：趋势图数据与 Excel 图表
    # ======================================================
    chart_sheet = workbook.create_sheet("趋势图数据")

    chart_headers = [
        "时间（秒）",
        "面部活跃度",
        "观看投入度",
        "紧张相关指数"
    ]

    for column, header in enumerate(
        chart_headers,
        start=1
    ):
        cell = chart_sheet.cell(
            row=1,
            column=column,
            value=header
        )

        cell.fill = title_fill
        cell.font = Font(
            bold=True,
            color="FFFFFF"
        )

    for row, record in enumerate(
        records,
        start=2
    ):
        chart_sheet.cell(
            row=row,
            column=1,
            value=record["time_seconds"]
        )

        chart_sheet.cell(
            row=row,
            column=2,
            value=record["activity"]
        )

        chart_sheet.cell(
            row=row,
            column=3,
            value=record["attention"]
        )

        chart_sheet.cell(
            row=row,
            column=4,
            value=record["tension"]
        )

    line_chart = LineChart()
    line_chart.title = "面部行为趋势"
    line_chart.y_axis.title = "相对指数"
    line_chart.x_axis.title = "时间（秒）"
    line_chart.height = 10
    line_chart.width = 20

    data = Reference(
        chart_sheet,
        min_col=2,
        max_col=4,
        min_row=1,
        max_row=len(records) + 1
    )

    categories = Reference(
        chart_sheet,
        min_col=1,
        min_row=2,
        max_row=len(records) + 1
    )

    line_chart.add_data(
        data,
        titles_from_data=True
    )

    line_chart.set_categories(categories)

    chart_sheet.add_chart(
        line_chart,
        "F2"
    )

    workbook.save(file_path)


def generate_reports(
    session_id,
    records,
    checkpoints,
    suggestions
):
    """
    一次测量结束后调用此函数，
    生成 CSV、PNG、TXT、Excel 四份文件。
    """

    if not records:
        return None

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    base_name = f"{session_id}_{timestamp}"

    csv_path = OUTPUT_DIR / f"{base_name}.csv"
    chart_path = OUTPUT_DIR / f"{base_name}_trend.png"
    report_path = OUTPUT_DIR / f"{base_name}_report.txt"
    excel_path = OUTPUT_DIR / f"{base_name}_summary.xlsx"

    save_csv(records, csv_path)

    save_chart(records, chart_path)

    save_text_report(
        report_path,
        checkpoints,
        records,
        suggestions
    )

    save_excel(
        excel_path,
        checkpoints,
        records,
        suggestions
    )

    return {
        "csv_file": csv_path.name,
        "chart_file": chart_path.name,
        "report_file": report_path.name,
        "excel_file": excel_path.name
    }