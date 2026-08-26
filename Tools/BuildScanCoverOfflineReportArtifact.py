#!/usr/bin/env python3
"""Build the canonical Data Analytics artifact for the ScanCover offline audit."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison_report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def by_name(items: list[dict]) -> dict[str, dict]:
    return {item["name"]: item for item in items}


def pct(value: float) -> str:
    return f"{value:.1f}%"


def materialize_sqlite(
    path: Path,
    datasets: dict[str, list[dict]],
    queries: dict[str, str],
) -> dict[str, list[dict]]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        for table, rows in datasets.items():
            if not rows:
                raise RuntimeError(f"Cannot materialize empty dataset: {table}")
            columns = list(rows[0].keys())
            types: list[str] = []
            for column in columns:
                sample = next((row[column] for row in rows if row[column] is not None), None)
                if isinstance(sample, bool) or isinstance(sample, int):
                    types.append("INTEGER")
                elif isinstance(sample, float):
                    types.append("REAL")
                else:
                    types.append("TEXT")
            connection.execute(f'DROP TABLE IF EXISTS "{table}"')
            definition = ", ".join(f'"{column}" {kind}' for column, kind in zip(columns, types))
            connection.execute(f'CREATE TABLE "{table}" ({definition})')
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f'INSERT INTO "{table}" VALUES ({placeholders})',
                [[row[column] for column in columns] for row in rows],
            )
        connection.commit()
        output: dict[str, list[dict]] = {}
        for name, query in queries.items():
            output[name] = [dict(row) for row in connection.execute(query).fetchall()]
        return output
    finally:
        connection.close()


def main() -> int:
    args = parse_args()
    report = json.loads(args.comparison_report.read_text(encoding="utf-8-sig"))
    comparisons = by_name(report["comparisons"])
    chain = by_name(report["responsibilityChain"]["buckets"])
    candidate_groups = by_name(report["stableCandidateToPaper"]["qualityByPaperProximity"])
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    coverage_rows = [
        {
            "stage": "清洗后深度",
            "coverage_rate": comparisons["post_qrs"]["systemCoveredWithin10cmPct"] / 100.0,
            "p50_mm": comparisons["post_qrs"]["sourceToSystemMeters"]["p50"] * 1000.0,
            "p95_mm": comparisons["post_qrs"]["sourceToSystemMeters"]["p95"] * 1000.0,
            "source_points": comparisons["post_qrs"]["pointCount"],
        },
        {
            "stage": "稳定枪胶候选",
            "coverage_rate": comparisons["gungel_stable_candidates"]["systemCoveredWithin10cmPct"] / 100.0,
            "p50_mm": comparisons["gungel_stable_candidates"]["sourceToSystemMeters"]["p50"] * 1000.0,
            "p95_mm": comparisons["gungel_stable_candidates"]["sourceToSystemMeters"]["p95"] * 1000.0,
            "source_points": comparisons["gungel_stable_candidates"]["pointCount"],
        },
        {
            "stage": "纸皮前台",
            "coverage_rate": comparisons["paper_front"]["systemCoveredWithin10cmPct"] / 100.0,
            "p50_mm": comparisons["paper_front"]["sourceToSystemMeters"]["p50"] * 1000.0,
            "p95_mm": comparisons["paper_front"]["sourceToSystemMeters"]["p95"] * 1000.0,
            "source_points": comparisons["paper_front"]["pointCount"],
        },
    ]
    responsibility_labels = {
        "paper_published": "纸皮已发布",
        "paper_missing_despite_stable_candidate": "有稳定候选但无纸皮",
        "stable_candidate_missing_despite_post_qrs_depth": "有深度但无稳定候选",
        "post_qrs_depth_absent_or_external_reference_disagrees": "深度未覆盖或外参照不一致",
    }
    responsibility_rows = [
        {
            "responsibility": responsibility_labels[name],
            "share_rate": item["sharePct"] / 100.0,
            "sample_count": item["count"],
            "threshold_cm": report["responsibilityChain"]["thresholdMeters"] * 100.0,
        }
        for name, item in chain.items()
    ]
    sensitivity_rows = []
    for item in report["responsibilityChainSensitivity"]:
        for bucket in item["buckets"]:
            sensitivity_rows.append(
                {
                    "threshold_cm": item["thresholdMeters"] * 100.0,
                    "responsibility": responsibility_labels[bucket["name"]],
                    "share_rate": bucket["sharePct"] / 100.0,
                    "sample_count": bucket["count"],
                }
            )

    candidate_labels = {
        "paper_within_10cm": "纸皮在10cm内",
        "paper_10_to_20cm": "纸皮距10-20cm",
        "paper_over_20cm": "纸皮距20cm外",
    }
    candidate_rows = []
    for name, group in candidate_groups.items():
        metrics = group["metrics"]
        candidate_rows.append(
            {
                "paper_proximity": candidate_labels[name],
                "candidate_count": group["count"],
                "share_rate": group["sharePct"] / 100.0,
                "observations_p50": metrics["observation_count"]["p50"],
                "sigma_mm_p50": metrics["sigma_mm"]["p50"],
                "effective_support_p50": metrics["effective_support"]["p50"],
                "dual_agree_p50": metrics["dual_agree_support"]["p50"],
                "opposition_p95": metrics["opposition_votes"]["p95"],
                "view_spread_deg_p50": metrics["view_spread_deg"]["p50"],
            }
        )

    fov = report["postQrsStrata"]["byFieldPosition"]
    motion = report["postQrsStrata"]["byMotion"]
    strata_rows = [
        {
            "dimension": "视野位置",
            "bucket": label,
            "sample_count": fov[name]["count"],
            "p95_mm": fov[name]["distanceMeters"]["p95"] * 1000.0,
            "within_5cm_rate": fov[name]["within5cmPct"] / 100.0,
            "over_20cm_rate": fov[name]["over20cmPct"] / 100.0,
        }
        for name, label in (("center", "中心"), ("middle", "中带"), ("edge", "边缘"))
    ] + [
        {
            "dimension": "运动",
            "bucket": label,
            "sample_count": motion[name]["count"],
            "p95_mm": motion[name]["distanceMeters"]["p95"] * 1000.0,
            "within_5cm_rate": motion[name]["within5cmPct"] / 100.0,
            "over_20cm_rate": motion[name]["over20cmPct"] / 100.0,
        }
        for name, label in (("still", "静止"), ("medium", "中速"), ("fast", "快速"))
    ]

    headline = [{
        "post_depth_coverage": coverage_rows[0]["coverage_rate"],
        "stable_candidate_coverage": coverage_rows[1]["coverage_rate"],
        "paper_coverage": coverage_rows[2]["coverage_rate"],
        "paper_existing_p95_mm": coverage_rows[2]["p95_mm"],
        "registration_fitness": report["registration"]["fineFitness"],
        "registration_rmse_mm": report["registration"]["fineInlierRmseMeters"] * 1000.0,
    }]

    sql_queries = {
        "headline": "SELECT * FROM headline;",
        "stage_coverage": "SELECT * FROM stage_coverage ORDER BY CASE stage WHEN '清洗后深度' THEN 1 WHEN '稳定枪胶候选' THEN 2 ELSE 3 END;",
        "responsibility": "SELECT * FROM responsibility ORDER BY share_rate DESC;",
        "sensitivity": "SELECT * FROM sensitivity ORDER BY threshold_cm, responsibility;",
        "candidate_quality": "SELECT * FROM candidate_quality ORDER BY share_rate DESC;",
        "strata": "SELECT * FROM strata ORDER BY dimension, p95_mm DESC;",
    }
    materialized = materialize_sqlite(
        args.out.parent / "offline_report.sqlite",
        {
            "headline": headline,
            "stage_coverage": coverage_rows,
            "responsibility": responsibility_rows,
            "sensitivity": sensitivity_rows,
            "candidate_quality": candidate_rows,
            "strata": strata_rows,
        },
        sql_queries,
    )
    headline = materialized["headline"]
    coverage_rows = materialized["stage_coverage"]
    responsibility_rows = materialized["responsibility"]
    sensitivity_rows = materialized["sensitivity"]
    candidate_rows = materialized["candidate_quality"]
    strata_rows = materialized["strata"]

    system_points = report["responsibilityChain"]["pointCount"]
    paper_accuracy = comparisons["paper_front"]
    depth_filter = report["depthFilterSummary"]
    title = "ScanCover 离线责任链审计"
    source_id = "offline_compare_output"

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Quest 3 同一房间的回放会话、枪胶候选、纸皮前台与系统房间网格的离线对照。",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "post_depth_coverage_card",
                    "description": "系统参照表面在10 cm内能找到清洗后深度证据的比例。",
                    "dataset": "headline",
                    "sourceId": "headline_sql",
                    "metrics": [{"label": "清洗后深度覆盖", "field": "post_depth_coverage", "format": "percent"}],
                },
                {
                    "id": "candidate_coverage_card",
                    "description": "系统参照表面在10 cm内能找到稳定枪胶候选的比例。",
                    "dataset": "headline",
                    "sourceId": "headline_sql",
                    "metrics": [{"label": "稳定候选覆盖", "field": "stable_candidate_coverage", "format": "percent"}],
                },
                {
                    "id": "paper_coverage_card",
                    "description": "系统参照表面在10 cm内能找到纸皮前台几何的比例。",
                    "dataset": "headline",
                    "sourceId": "headline_sql",
                    "metrics": [{"label": "纸皮覆盖", "field": "paper_coverage", "format": "percent"}],
                },
                {
                    "id": "paper_p95_card",
                    "description": "已发布纸皮采样点到系统参照表面的第95百分位距离。",
                    "dataset": "headline",
                    "sourceId": "headline_sql",
                    "metrics": [{"label": "已有纸皮 p95", "field": "paper_existing_p95_mm", "format": "number", "unit": "mm"}],
                },
            ],
            "charts": [
                {
                    "id": "stage_coverage_chart",
                    "title": "各处理层对系统参照的10 cm覆盖率",
                    "subtitle": "同一刚体配准、系统表面均匀采样40万点；覆盖在深度之后连续下降。",
                    "type": "bar",
                    "dataset": "stage_coverage",
                    "sourceId": "stage_coverage_sql",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "stage", "type": "nominal", "label": "处理层"},
                        "y": {"field": "coverage_rate", "type": "quantitative", "label": "覆盖率", "format": "percent"},
                        "tooltip": [
                            {"field": "p50_mm", "type": "quantitative", "label": "点到参照 p50 (mm)"},
                            {"field": "p95_mm", "type": "quantitative", "label": "点到参照 p95 (mm)"},
                            {"field": "source_points", "type": "quantitative", "label": "采样点数"},
                        ],
                    },
                },
                {
                    "id": "responsibility_chart",
                    "title": "10 cm判据下的互斥责任分解",
                    "subtitle": "每个系统参照采样点只进入一个最下游断点，四项严格合计100%。",
                    "type": "bar",
                    "dataset": "responsibility",
                    "sourceId": "responsibility_sql",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "responsibility", "type": "nominal", "label": "最下游状态"},
                        "y": {"field": "share_rate", "type": "quantitative", "label": "表面占比", "format": "percent"},
                        "tooltip": [
                            {"field": "sample_count", "type": "quantitative", "label": "参照点数"},
                            {"field": "threshold_cm", "type": "quantitative", "label": "距离阈值 (cm)"},
                        ],
                    },
                },
                {
                    "id": "sensitivity_chart",
                    "title": "责任分解对距离阈值的灵敏度",
                    "subtitle": "5-20 cm；10 cm以上时“有稳定候选但无纸皮”始终是最大未完成项。",
                    "type": "bar",
                    "dataset": "sensitivity",
                    "sourceId": "sensitivity_sql",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "threshold_cm", "type": "quantitative", "label": "距离阈值 (cm)"},
                        "y": {"field": "share_rate", "type": "quantitative", "label": "表面占比", "format": "percent"},
                        "color": {"field": "responsibility", "type": "nominal", "label": "责任状态"},
                        "tooltip": [{"field": "sample_count", "type": "quantitative", "label": "参照点数"}],
                    },
                },
            ],
            "tables": [
                {
                    "id": "candidate_quality_table",
                    "title": "稳定候选按纸皮距离分组",
                    "subtitle": "未被纸皮覆盖的候选仍有大量观测和双证词，反对票第95百分位为0。",
                    "dataset": "candidate_quality",
                    "sourceId": "candidate_quality_sql",
                    "defaultSort": {"field": "share_rate", "direction": "desc"},
                    "columns": [
                        {"field": "paper_proximity", "label": "纸皮距离", "type": "text"},
                        {"field": "candidate_count", "label": "稳定候选数", "format": "number"},
                        {"field": "share_rate", "label": "候选占比", "format": "percent"},
                        {"field": "observations_p50", "label": "观测数 p50", "format": "number"},
                        {"field": "sigma_mm_p50", "label": "sigma p50 (mm)", "format": "number"},
                        {"field": "effective_support_p50", "label": "有效支撑 p50", "format": "number"},
                        {"field": "dual_agree_p50", "label": "双证词 p50", "format": "number"},
                        {"field": "opposition_p95", "label": "反对票 p95", "format": "number"},
                    ],
                },
                {
                    "id": "strata_table",
                    "title": "清洗后深度的视野与运动分桶",
                    "subtitle": "视野边缘尾差明显增大；本次样本中快速运动不是总体误差的主导项。",
                    "dataset": "strata",
                    "sourceId": "strata_sql",
                    "defaultSort": {"field": "p95_mm", "direction": "desc"},
                    "columns": [
                        {"field": "dimension", "label": "维度", "type": "text"},
                        {"field": "bucket", "label": "分桶", "type": "text"},
                        {"field": "sample_count", "label": "采样点数", "format": "number"},
                        {"field": "p95_mm", "label": "到参照 p95 (mm)", "format": "number"},
                        {"field": "within_5cm_rate", "label": "5 cm内", "format": "percent"},
                        {"field": "over_20cm_rate", "label": "20 cm外", "format": "percent"},
                    ],
                },
            ],
            "sources": [
                {"id": source_id, "label": "离线比较结果", "path": "derived/comparison_report.json"},
                {"id": "headline_sql", "label": "头部指标查询", "path": "derived/offline_report.sqlite"},
                {"id": "stage_coverage_sql", "label": "处理层覆盖查询", "path": "derived/offline_report.sqlite"},
                {"id": "responsibility_sql", "label": "互斥责任查询", "path": "derived/offline_report.sqlite"},
                {"id": "sensitivity_sql", "label": "阈值灵敏度查询", "path": "derived/offline_report.sqlite"},
                {"id": "candidate_quality_sql", "label": "候选质量查询", "path": "derived/offline_report.sqlite"},
                {"id": "strata_sql", "label": "视野与运动分桶查询", "path": "derived/offline_report.sqlite"},
                {"id": "replay_session", "label": "独立回放会话", "path": "session/replay_contract.json"},
                {"id": "system_room_mesh", "label": "Quest系统房间网格", "path": "system/meta_scene_mesh_aligned_all.obj"},
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": source_id,
                    "body": (
                        "## 技术结论：离线已经可用，当前最大断点在纸皮发布\n\n"
                        f"- **底层不是主要缺数。** 清洗后深度对系统参照的10 cm覆盖为 "
                        f"{pct(coverage_rows[0]['coverage_rate'] * 100.0)}，稳定候选为 "
                        f"{pct(coverage_rows[1]['coverage_rate'] * 100.0)}，纸皮仅 "
                        f"{pct(coverage_rows[2]['coverage_rate'] * 100.0)}。\n"
                        f"- **纸皮已有部分偏保守但几何较准。** 已发布纸皮到系统参照的 p95 为 "
                        f"{paper_accuracy['sourceToSystemMeters']['p95'] * 1000.0:.1f} mm，"
                        f"{paper_accuracy['sourceWithin10cmPct']:.1f}% 的纸皮采样点在10 cm内。\n"
                        f"- **责任链中最大的未完成项是“已有稳定候选但没有纸皮”**，占系统参照表面 "
                        f"{chain['paper_missing_despite_stable_candidate']['sharePct']:.1f}%；"
                        f"其次是“已有深度但没有稳定候选” {chain['stable_candidate_missing_despite_post_qrs_depth']['sharePct']:.1f}%。\n"
                        "- 这轮结果支持：下一刀先查公共外皮合成器的闭合/一致性发布契约，再查深度到候选的稀疏化；"
                        "不应继续在显示材质或HERA页上补覆盖。"
                    ),
                },
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["post_depth_coverage_card", "candidate_coverage_card", "paper_coverage_card", "paper_p95_card"]},
                {
                    "id": "coverage_finding",
                    "type": "markdown",
                    "sourceId": source_id,
                    "body": (
                        "## 证据在向下游递减，而不是深度层先天看不见\n\n"
                        "下图对三层使用完全相同的系统参照表面、刚体变换和10 cm判据。"
                        "清洗后深度覆盖接近九成，但到稳定候选损失约28个百分点，再到纸皮再损失约19个百分点。"
                        "因此画面空洞不能再笼统归咎于Quest深度缺失。"
                    ),
                },
                {"id": "stage_coverage", "type": "chart", "chartId": "stage_coverage_chart", "layout": "full"},
                {
                    "id": "responsibility_finding",
                    "type": "markdown",
                    "sourceId": source_id,
                    "body": (
                        "## 互斥归责把“空洞”拆成了四种来源\n\n"
                        f"40万系统表面采样点严格只进入一个桶：纸皮已发布 {chain['paper_published']['sharePct']:.1f}%，"
                        f"有稳定候选但无纸皮 {chain['paper_missing_despite_stable_candidate']['sharePct']:.1f}%，"
                        f"有清洗后深度但无稳定候选 {chain['stable_candidate_missing_despite_post_qrs_depth']['sharePct']:.1f}%，"
                        f"深度未覆盖或外参照不一致 {chain['post_qrs_depth_absent_or_external_reference_disagrees']['sharePct']:.1f}%。"
                        "这把第一责任从原料层移到了纸皮拓扑发布契约。"
                    ),
                },
                {"id": "responsibility", "type": "chart", "chartId": "responsibility_chart", "layout": "full"},
                {
                    "id": "candidate_finding",
                    "type": "markdown",
                    "sourceId": source_id,
                    "body": (
                        "## 未出纸的稳定候选并非普遍缺证据\n\n"
                        f"{candidate_groups['paper_over_20cm']['count']} 个稳定候选在20 cm内找不到纸皮；"
                        f"它们的观测数中位数仍为 {candidate_groups['paper_over_20cm']['metrics']['observation_count']['p50']:.0f}，"
                        f"双证词中位数 {candidate_groups['paper_over_20cm']['metrics']['dual_agree_support']['p50']:.0f}，"
                        "反对票 p95 为0。它们的有效支撑弱于已发布组，但远非“没有数据”。"
                        "结合当前着色器代码，最可疑的是稳定点进入纸面后仍必须凑齐单元闭合轮廓、同层与几何一致性，"
                        "单点稳定不能直接获得发布资格。"
                    ),
                },
                {"id": "candidate_quality", "type": "table", "tableId": "candidate_quality_table", "layout": "full"},
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "body": (
                        "## 数据范围与口径\n\n"
                        f"本轮使用一次约84秒的右眼独立回放会话：1400组平台原始/清洗后同帧深度、1030条已接受融合输入、"
                        f"3354个稳定枪胶候选、最终纸皮前台和另一次启动导出的Quest系统房间网格。"
                        f"系统网格均匀采样 {system_points:,} 点。\n\n"
                        "“覆盖”定义为：对系统参照表面上的每个采样点，在指定处理层10 cm内至少找到一个点或三角面采样。"
                        "它衡量空间证据能否到达该层，不等同于三角形拓扑完美，也不把系统网格定义为物理真值。"
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## 方法：一次配准，所有层共用\n\n"
                        "从深度纹理按记录的右眼投影逆矩阵与视图逆矩阵反投影；每5帧取一帧、像素步长6。"
                        "仅用清洗后深度估计ScanCover世界到系统网格的刚体变换，不允许缩放；"
                        "从12个水平旋转假设中选择粗配准，再做点到平面ICP。"
                        f"最终匹配率 {report['registration']['fineFitness']:.4f}，匹配内RMSE "
                        f"{report['registration']['fineInlierRmseMeters'] * 1000.0:.1f} mm。"
                        "同一个变换随后原封不动用于平台原始深度、稳定候选和纸皮，因此层间差异可比较。"
                    ),
                },
                {
                    "id": "robustness",
                    "type": "markdown",
                    "sourceId": source_id,
                    "body": (
                        "## 阈值复核与仍需保留的不确定性\n\n"
                        "5 cm阈值低于本次跨启动配准RMSE附近，容易把配准残差误判成层内缺失；主结论采用10 cm。"
                        "把阈值放宽到15或20 cm后，“有稳定候选但无纸皮”仍是最大的未完成桶，"
                        "说明纸皮发布断点不是10 cm偶然切出来的。\n\n"
                        "系统房间网格是外部参照而非绝对真值；它可能抹平家具细节、缺少局部结构，也没有和本次扫描同一时刻的位姿记录。"
                        "因此本报告能证明层间信息在哪里丢失，不能仅凭系统差异判定哪一侧的每个局部深度物理正确。"
                    ),
                },
                {"id": "sensitivity", "type": "chart", "chartId": "sensitivity_chart", "layout": "full"},
                {
                    "id": "view_motion",
                    "type": "markdown",
                    "sourceId": source_id,
                    "body": (
                        "## 视野边缘仍有影响，速度不是本次总体空洞的主因\n\n"
                        f"清洗后深度在视野中心的 p95 为 {fov['center']['distanceMeters']['p95'] * 1000.0:.1f} mm，"
                        f"边缘为 {fov['edge']['distanceMeters']['p95'] * 1000.0:.1f} mm；边缘尾差确实更大。"
                        f"快速运动 p95 为 {motion['fast']['distanceMeters']['p95'] * 1000.0:.1f} mm，"
                        f"静止为 {motion['still']['distanceMeters']['p95'] * 1000.0:.1f} mm，差异仅约 "
                        f"{(motion['fast']['distanceMeters']['p95'] - motion['still']['distanceMeters']['p95']) * 1000.0:.1f} mm。"
                        "速度保护罩仍有抑制极端点浪的价值，但不能解释从89%深度覆盖掉到43%纸皮覆盖。"
                    ),
                },
                {"id": "strata", "type": "table", "tableId": "strata_table", "layout": "full"},
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 下一步应验证发布契约，而不是继续修显示\n\n"
                        "1. 在离线侧复刻公共外皮合成器的单元闭合裁决，给每个“有稳定候选但无纸皮”的位置记录失败原因："
                        "交叉不足、面上1/3条边、跨层、刀片、法线不支持或同拓扑几何不一致。\n"
                        "2. 用同一回放做一个只改变拓扑发布契约的A/B：保留稳定候选和反对票不变，比较覆盖、错误面和非流形率。\n"
                        "3. 第二优先级才是深度到候选：查20.7%的‘有深度无稳定候选’是否集中在候选重归属、邻域反对票或支持阈值。\n"
                        "4. 暂不继续给HERA、材质或显示层添加补洞规则；它们会掩盖而不会修复这两个断点。"
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 尚未闭环的问题\n\n"
                        "- 本轮还没有在CPU离线重演完整的VolumeIntegration.compute与SupportTruthExtract.compute；"
                        "因此已定性到‘哪一层丢’，尚未把25.9%逐单元分摊到每一个着色器拒绝分支。\n"
                        "- 需要确认同一房间再采一次、扫描轨迹变化后，责任排序是否保持；若保持，才适合进入生产契约修改。\n"
                        f"- 预处理移除了 {depth_filter['rawValidRemovedPct']:.2f}% 的原始有效像素，"
                        f"原始/清洗后深度差 p95 为 {depth_filter['absRawPostDiffMm']['p95']:.1f} mm；"
                        "它改善尾差但也略减覆盖，后续可在责任链闭环后单独评估。"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "stage_coverage": coverage_rows,
                "responsibility": responsibility_rows,
                "sensitivity": sensitivity_rows,
                "candidate_quality": candidate_rows,
                "strata": strata_rows,
            },
        },
        "sources": [
            {
                "id": source_id,
                "query": {
                    "engine": "python",
                    "language": "python",
                    "description": "Rigid registration, nearest-surface distances, stage coverage and mutually exclusive responsibility decomposition.",
                    "tables_used": ["comparison_report.json"],
                    "filters": [
                        "depth frame stride 5",
                        "depth pixel stride 6",
                        "main coverage threshold 0.10 metres",
                        "rigid transform only, no scale",
                    ],
                    "metric_definitions": {
                        "coverage_rate": "uniform system-mesh sample points with a nearest layer point or sampled face within the stated threshold divided by all system samples",
                        "responsibility_share": "mutually exclusive downstream-most state divided by all 400000 system samples",
                    },
                    "executed_at": generated_at,
                },
            },
            *[
                {
                    "id": f"{name}_sql",
                    "query": {
                        "engine": "sqlite",
                        "language": "sql",
                        "sql": query,
                        "description": f"Reads the reviewed {name} rows materialized by the offline comparison report builder.",
                        "tables_used": [name],
                        "executed_at": generated_at,
                    },
                }
                for name, query in sql_queries.items()
            ],
            {
                "id": "replay_session",
                "query": {
                    "engine": "filesystem",
                    "description": "Self-contained ScanCover replay session and captured production artifacts.",
                    "tables_used": ["depth_pairs/manifest.csv", "gungel_candidate_audit/candidates.csv", "paper_audit/paper_front.ply"],
                },
            },
            {
                "id": "system_room_mesh",
                "query": {
                    "engine": "filesystem",
                    "description": "Quest system room mesh exported in a separate app launch and used only as an external comparison reference.",
                    "tables_used": ["meta_scene_mesh_aligned_all.obj"],
                },
            },
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    notes = {
        "audience": "technical",
        "requiredStructure": [
            "title",
            "technical summary",
            "key findings with visual evidence",
            "scope data and metric definitions",
            "methodology",
            "limitations uncertainty and robustness checks",
            "recommended next steps",
            "further questions",
        ],
        "chartMap": [
            {
                "section": "evidence decreases downstream",
                "question": "where does coverage disappear",
                "family": "comparison",
                "type": "bar",
                "fields": ["stage", "coverage_rate"],
                "claim": "coverage falls mainly after post-QRS depth",
                "palette": "single-root preferred",
            },
            {
                "section": "mutually exclusive responsibility",
                "question": "which stage owns each missing region",
                "family": "composition comparison",
                "type": "bar",
                "fields": ["responsibility", "share_rate"],
                "claim": "paper publication is the largest unresolved stage at 10 cm",
                "palette": "single-root preferred",
            },
            {
                "section": "threshold robustness",
                "question": "does the stage ranking depend on one threshold",
                "family": "comparison",
                "type": "grouped bar",
                "fields": ["threshold_cm", "share_rate", "responsibility"],
                "claim": "paper publication remains the largest incomplete bucket from 10 to 20 cm",
                "palette": "relaxed multi-category",
            },
        ],
        "omissions": [
            "No causal claim against the Quest system mesh because it is a separate-launch external reference.",
            "No spatial screenshot chart; the colored PLY files preserve the three-dimensional evidence at full analytical grain.",
        ],
    }
    (args.out.parent / "report_source_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
