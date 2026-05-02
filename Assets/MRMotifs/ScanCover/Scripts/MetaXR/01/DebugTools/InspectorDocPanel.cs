using System;
using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class InspectorDocPanel : MonoBehaviour
{
    public enum PresetType
    {
        None,
        SessionReferenceFrame,
        ScanWorkCenter,
        SurfaceSnapPoint
    }

    public enum SectionStyle
    {
        Note,
        Info,
        Warning,
        Tip,
        Checklist,
        Flow,
        Custom
    }

    [Serializable]
    public class DocSection
    {
        public bool enabled = true;
        public bool expanded = true;

        public SectionStyle style = SectionStyle.Info;
        public string title = "Section";

        [TextArea(2, 8)]
        public string body = "";

        [Tooltip("可选：用于显示额外标签，例如 Input / Output / Pitfall")]
        public string tag = "";

        [Tooltip("可选：外部文档链接（Notion/GitHub/Wiki等）")]
        public string url = "";
    }

    [Header("Panel")]
    public bool panelEnabled = true;

    public PresetType presetType = PresetType.None;

    public string panelTitle = "Inspector Notes";

    [TextArea(1, 4)]
    public string panelSummary = "";

    [Tooltip("显示时是否高亮顶部摘要")]
    public bool emphasizeSummary = true;

    [Header("Sections")]
    public List<DocSection> sections = new List<DocSection>();

    [Header("Optional Links")]
    public string quickLinkLabel = "Open Docs";
    public string quickLinkUrl = "";

    [Header("Debug")]
    public bool showInPlayMode = true;

    private void Reset()
    {
        if (sections == null) sections = new List<DocSection>();
        if (sections.Count == 0)
        {
            sections.Add(new DocSection
            {
                style = SectionStyle.Info,
                title = "用途",
                body = "在这里填写该对象的职责、上下游、调试要点。",
                tag = "Purpose"
            });
        }
    }

    [ContextMenu("Apply Preset Template")]
    public void ApplyPresetTemplate()
    {
        ApplyPreset(presetType);
    }

    public void ApplyPreset(PresetType preset)
    {
        sections ??= new List<DocSection>();
        sections.Clear();

        switch (preset)
        {
            case PresetType.SessionReferenceFrame:
                panelTitle = "SessionReferenceFrame";
                panelSummary = "稳定参考系根节点：定义系统统一坐标基准，用于承载 ScanWorkCenter 与 SurfaceSnapPoint。";

                sections.Add(new DocSection
                {
                    style = SectionStyle.Info,
                    title = "职责",
                    tag = "Role",
                    body = "给扫描/覆盖系统提供统一坐标系；后续累计、转换、对齐都应基于该参考系。"
                });
                sections.Add(new DocSection
                {
                    style = SectionStyle.Warning,
                    title = "不负责什么",
                    tag = "Not For",
                    body = "不负责表面吸附，不负责扫描特效，不应跟随头显移动。"
                });
                sections.Add(new DocSection
                {
                    style = SectionStyle.Checklist,
                    title = "检查项",
                    tag = "Checklist",
                    body = "1) 位置稳定\n2) 不跟随头显\n3) 作为 ScanWorkCenter / SurfaceSnapPoint 的父节点"
                });
                break;

            case PresetType.ScanWorkCenter:
                panelTitle = "ScanWorkCenter";
                panelSummary = "扫描/覆盖调度中心（逻辑点）：稳定优先，不要求贴真实表面。";

                sections.Add(new DocSection
                {
                    style = SectionStyle.Info,
                    title = "职责",
                    tag = "Role",
                    body = "作为扫描区域中心、chunk 更新中心、预算调度中心；为 SurfaceSnapPoint 提供寻表面起点。"
                });
                sections.Add(new DocSection
                {
                    style = SectionStyle.Warning,
                    title = "常见误解",
                    tag = "Pitfall",
                    body = "它不是表面吸附点。黄色 marker 不贴地通常是正常现象。"
                });
                sections.Add(new DocSection
                {
                    style = SectionStyle.Flow,
                    title = "数据流",
                    tag = "Flow",
                    body = "SessionReferenceFrame -> ScanWorkCenterFollower -> ScanWorkCenter -> ScanSurfaceSnapper"
                });
                break;

            case PresetType.SurfaceSnapPoint:
                panelTitle = "SurfaceSnapPoint / ScanSurfaceSnapper";
                panelSummary = "真实表面吸附点（物理点）：通过环境射线命中真实表面，用于贴面视觉与表面法线。";

                sections.Add(new DocSection
                {
                    style = SectionStyle.Info,
                    title = "职责",
                    tag = "Role",
                    body = "提供真实表面命中点（位置）和表面法线（方向），用于扫描冲击波、贴面刷写、视觉落点。"
                });
                sections.Add(new DocSection
                {
                    style = SectionStyle.Tip,
                    title = "调试提示",
                    tag = "Tip",
                    body = "绿色 marker 贴地灵敏是预期行为；命中失败要检查回退策略、射线起点高度、法线过滤。"
                });
                sections.Add(new DocSection
                {
                    style = SectionStyle.Warning,
                    title = "限制",
                    tag = "Limit",
                    body = "命中受环境深度视锥限制，不能把它当成全局稳定调度中心。"
                });
                break;

            case PresetType.None:
            default:
                if (string.IsNullOrWhiteSpace(panelTitle))
                    panelTitle = "Inspector Notes";
                if (sections.Count == 0)
                {
                    sections.Add(new DocSection
                    {
                        style = SectionStyle.Info,
                        title = "Section",
                        body = "填写说明内容…"
                    });
                }
                break;
        }
    }
}