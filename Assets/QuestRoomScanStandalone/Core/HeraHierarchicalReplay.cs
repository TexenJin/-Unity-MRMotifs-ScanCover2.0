using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using Unity.Mathematics;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// HERA (Hierarchical Evidence Routing and Atomic replacement) read-only
    /// replay.  64-cells own identity/lifetime only, 32-cells route evidence,
    /// and 16-cells are created only below unresolved 32-cell pages.
    /// </summary>
    internal sealed class HeraHierarchicalReplay : IDisposable
    {
        // One row is one already classified child-page triangle bucket.  The
        // replay cannot compare triangle indices across 32/16 resolutions, but
        // these counters are produced by per-triangle tests inside the 16-cell
        // extractor.  Keeping the rows separate prevents an unrelated hard
        // conflict elsewhere in the family from vetoing an interior false
        // conflict that has its own local evidence.
        private sealed class TriangleCounterfactualPage
        {
            public int3 Coordinate;
            public int SuspectedCenterFalse;
            public int PureBoundary;
            public int RealEdge;
            public int AmbiguousNormal;
            public int AmbiguousWeight;
            public int AmbiguousMulti;
            public int Transition;
            public int InteriorRecoverable;
            public int InteriorShadowTriangles;
            public int BoundaryRecoverable;
            public int BoundaryShadowTriangles;
            public int BoundaryCommittedTriangles;
            public int BoundaryBlockedExactMissing;
            public int BoundaryShadowUnexpected;
            // Legacy family-wide counterfactual only. These values describe
            // what the former collective gate would have rejected; they do
            // not participate in exact-triangle production admission.
            public int BoundaryBlockedCoverage;
            public int BoundaryBlockedHard;
            public int BoundaryBlockedMulti;
            public int BoundaryBlockedTransition;
        }

        private sealed class Family
        {
            public int3 Parent;
            public int Ledger64Key;
            public int ParentKept;
            public int ParentDelegated;
            public int ExpectedChildren;
            public int CompletedChildren;
            public int ChildKept;
            public int ChildDelegated;
            public int ParentAmbiguousNormal;
            public int ParentAmbiguousWeight;
            public int ParentAmbiguousBoundary;
            public int ParentAmbiguousMulti;
            public int ParentRealEdge;
            public int ParentSuspectedCenterFalse;
            public int ParentTransition;
            public int ChildAmbiguousNormal;
            public int ChildAmbiguousWeight;
            public int ChildAmbiguousBoundary;
            public int ChildAmbiguousMulti;
            public int ChildRealEdge;
            public int ChildSuspectedCenterFalse;
            public int ChildTransition;
            public int AcceptedCoverageDelta;
            public int DelegatedReduction;
            public int BoundaryOnlyReduction;
            public int HardConflictDelta;
            public int TransitionDelta;
            public int SuspectedCenterFalseGain;
            public string DecisionReason = "pending";
            public string BoundaryCandidateReason = "pending";
            public bool BoundaryCandidatePass;
            public bool Finalized;
            public bool Swapped;
            public bool MixedCommitted;
            public int MixedCommittedPages;
            public int MixedCommittedTriangles;
            public readonly List<int3> Children = new List<int3>(8);
            public readonly List<TriangleCounterfactualPage> TrianglePages =
                new List<TriangleCounterfactualPage>(8);
            public int TriangleInteriorRecoverable;
            public int TriangleInteriorShadow;
            public int TrianglePureBoundaryCandidate;
            public int TriangleBoundaryRecoverable;
            public int TriangleBoundaryShadow;
            public int TriangleBoundaryCommitted;
            public int TriangleBoundaryBlockedExactMissing;
            public int TriangleBoundaryShadowUnexpected;
            public int BoundaryCommittedPages;
            public int TriangleBoundaryBlockedCoverage;
            public int TriangleBoundaryBlockedHard;
            public int TriangleBoundaryBlockedMulti;
            public int TriangleBoundaryBlockedTransition;
        }

        // 64^3 is deliberately a control-plane ledger only.  It records which
        // 32^3 families belong to the same persistent world region and how the
        // latest atomic replacement ended, but it never accepts/rejects mesh.
        private sealed class Ledger64Entry
        {
            public int3 Coordinate;
            public uint Version = 1;
            public int ParentPagesCompleted;
            public int ProblemParentPages;
            public long ParentKeptTriangles;
            public long ParentDelegatedTriangles;
            public int FamiliesQueued;
            public int FamiliesSwapped;
            public int FamiliesBlocked;
        }

        private readonly PersistentChunkMeshPipeline _parent32;
        private readonly PersistentChunkMeshPipeline _child16;
        private readonly Dictionary<int, Family> _families = new Dictionary<int, Family>();
        private readonly Dictionary<int, int> _childToParent = new Dictionary<int, int>();
        private readonly Dictionary<int, Ledger64Entry> _ledger64 =
            new Dictionary<int, Ledger64Entry>();
        private readonly int3 _parentGrid;
        private readonly int3 _childGrid;
        private readonly int3 _ledgerGrid;
        private bool _disposed;
        private bool _visible = true;
        /// <summary>当前是否绘制（帧率二分热键回读用；false 时提取/回读后台照跑）。</summary>
        public bool IsVisible => _visible;
        private bool _diagnosticColoring = true;
        private int _parentResults;
        private int _childResults;
        private int _familiesQueued;
        private int _familiesSwapped;
        private int _familiesBlocked;
        private int _familiesMixedCommitted;
        private int _mixedCommittedPages;
        private long _mixedCommittedTriangles;
        private int _boundaryCandidatePassed;
        private int _boundaryCandidateBlocked;
        private long _boundaryCandidateRecoverable;
        private long _parentKept;
        private long _parentDelegated;
        private long _parentDisplayRed;
        private long _parentConfirmationUnknown;
        private long _parentConfirmationPending;
        private long _parentConfirmationConfirmed;
        private long _parentConfirmationMixed;
        private long _childKept;
        private long _childDelegated;
        private long _parentInternalMixedContact;
        private long _parentInternalMixedRealEdge;
        private long _parentInternalMixedSuspectedCenterFalse;
        private long _parentInternalMixedAmbiguous;
        private long _parentDelegatedInternalMixedRealEdge;
        private long _parentDelegatedInternalMixedSuspectedCenterFalse;
        private long _parentDelegatedInternalMixedAmbiguous;
        private long _parentDelegatedTransition;
        private long _childInternalMixedContact;
        private long _childInternalMixedRealEdge;
        private long _childInternalMixedSuspectedCenterFalse;
        private long _childInternalMixedAmbiguous;
        private long _childDelegatedInternalMixedRealEdge;
        private long _childDelegatedInternalMixedSuspectedCenterFalse;
        private long _childDelegatedInternalMixedAmbiguous;
        private long _childDelegatedTransition;
        private long _triangleInteriorRecoverable;
        private long _triangleInteriorShadow;
        private long _trianglePureBoundaryCandidate;
        private long _triangleBoundaryRecoverable;
        private long _triangleBoundaryShadow;
        private long _triangleBoundaryCommitted;
        private long _triangleBoundaryBlockedExactMissing;
        private long _triangleBoundaryShadowUnexpected;
        private int _boundaryCommittedPages;
        private long _triangleBoundaryBlockedCoverage;
        private long _triangleBoundaryBlockedHard;
        private long _triangleBoundaryBlockedMulti;
        private long _triangleBoundaryBlockedTransition;
        private long _triangleRiskRealEdge;
        private long _triangleRiskNormal;
        private long _triangleRiskWeight;
        private long _triangleRiskMulti;
        private long _triangleRiskTransition;
        private int _drawValidationEarliestFrame = -1;
        private bool _drawValidationLogged;
        private bool _rescuePresentationUnlocked;

        // ── 增量精修模式（两段合一：边扫边按冻结块上屏）──
        // 与全场回放的差别：父页不自动排全场，由冻结调度器逐块喂入；没有"全场
        // 建完"的时刻，子页救回的全局绘制验证改为构造即解锁（逐页可见性仍由
        // FinalizeFamily 把关）；同一页可重提交（解冻重修），总账用逐页 tally
        // 先扣旧值防膨胀。
        private readonly bool _incrementalMode;
        private sealed class ParentTally
        {
            public long Kept, Delegated, DisplayRed;
            public long ConfUnknown, ConfPending, ConfConfirmed, ConfMixed;
            public long InternalContact, InternalRealEdge, InternalSuspected, InternalAmbiguous;
            public long DelegatedRealEdge, DelegatedSuspected, DelegatedAmbiguous, DelegatedTransition;
        }
        private readonly Dictionary<int, ParentTally> _parentTally =
            new Dictionary<int, ParentTally>();
        // 实时轨父页集合：这些页只出粗网格（tally 照记），提交时不建家族、
        // 不派生 16³ 子页——边界三角留给定稿轨（冻结后 QueueParentBlock 重
        // 提交时移出本集合，家族照常创建）补，省 8 倍子页负载。
        private readonly HashSet<int> _liveParentKeys = new HashSet<int>();

        public HeraHierarchicalReplay(
            VolumeIntegrator volume,
            ComputeShader compute,
            Material material,
            Transform parent,
            int layer,
            PersistentChunkMeshPipeline.Config parentConfig,
            PersistentChunkMeshPipeline.Config childConfig,
            Action<GPUSurfaceNets> extract,
            bool incrementalMode = false)
        {
            _incrementalMode = incrementalMode;
            _parent32 = new PersistentChunkMeshPipeline(
                volume, compute, material, parent, layer, parentConfig, extract);
            _child16 = new PersistentChunkMeshPipeline(
                volume, compute, material, parent, layer, childConfig, extract);
            _parentGrid = _parent32.ChunkGridCount;
            _childGrid = _child16.ChunkGridCount;
            _ledgerGrid = (_parentGrid + 1) / 2;

            _parent32.StaticReplayPageCommitted += OnParentCommitted;
            _child16.StaticReplayPageCommitted += OnChildCommitted;
            _parent32.SetDiagnosticColoring(true);
            _child16.SetDiagnosticColoring(true);
            _parent32.SetVisible(true);
            if (_incrementalMode)
            {
                // 增量模式没有"父页全场建完"的时刻：子页补丁可见性仍由
                // FinalizeFamily 逐家族把关，全局呈现锁在构造时直接解开。
                _rescuePresentationUnlocked = true;
                _child16.SetVisible(true);
            }
            else
            {
                // A child16 patch is meaningful only on top of the complete parent32
                // foundation.  Showing early child pages while the parent replay is
                // still being built creates misleading sparse rescue islands that
                // can look like a completed HERA result.
                _child16.SetVisible(false);
            }
        }

        public bool Failed => _parent32.Failed || _child16.Failed;
        public string FailureReason => _parent32.Failed
            ? _parent32.FailureReason
            : _child16.FailureReason;
        public int ParentBuilt => _parent32.BuiltChunkCount;
        public int ParentTotal => _parent32.ChunkCount;
        public int ChildBuilt => _childResults;
        public int ChildQueued => _childToParent.Count;
        public int Ledger64Count => _ledger64.Count;
        public int Ledger64Total => _ledgerGrid.x * _ledgerGrid.y * _ledgerGrid.z;
        public int FamiliesQueued => _familiesQueued;
        public int FamiliesSwapped => _familiesSwapped;
        public int FamiliesBlocked => _familiesBlocked;
        public int FamiliesFinalized => _familiesSwapped + _familiesBlocked;
        public int FamiliesPending => Math.Max(0, _familiesQueued - FamiliesFinalized);
        public int ChildrenPending => Math.Max(0, ChildQueued - ChildBuilt);
        public bool IsComplete =>
            !Failed &&
            ParentBuilt >= ParentTotal &&
            ChildBuilt >= ChildQueued &&
            FamiliesFinalized >= FamiliesQueued;
        // Vertex buffers contain an intentionally shared/copy-through payload,
        // so their count is storage, not a reliable draw statistic.
        public long StoredVertexPayload =>
            _parent32.AcceptedVertexCount + _child16.AcceptedVertexCount;
        public long VisibleTriangles
        {
            get
            {
                long visible = _parentKept;
                foreach (Family family in _families.Values)
                {
                    if (family.Swapped)
                    {
                        visible -= family.ParentKept;
                        visible += family.ChildKept;
                        visible += family.TriangleBoundaryCommitted;
                    }
                    else if (family.MixedCommitted)
                    {
                        visible += family.MixedCommittedTriangles;
                    }
                }
                return Math.Max(0L, visible);
            }
        }
        // Kept for the existing HUD API. This is explicitly storage payload,
        // while VisibleTriangles is the authoritative front-buffer draw count.
        public long VisibleVertices => StoredVertexPayload;

        public string CompactStats =>
            $"32:{ParentBuilt}/{ParentTotal} 16:{ChildBuilt}/{ChildQueued} " +
            $"换:{FamiliesSwapped} 保:{FamiliesBlocked} 64账:{Ledger64Count}/{Ledger64Total}";

        // 双色画面保持不变；红色现在只表达确认尚未闭环。出生来源混合仍可
        // 请求 16 级复核，但不再单独把已经真实确认的三角染红。
        public string RedCauseCompactStats =>
            $"红{_parentDisplayRed} 未知{_parentConfirmationUnknown} " +
            $"待证{_parentConfirmationPending} 混证{_parentConfirmationMixed} " +
            $"已证绿{_parentConfirmationConfirmed} 16复核{_parentDelegated}";

        public void Tick()
        {
            if (_disposed) return;
            _parent32.Tick();
            _child16.Tick();
            // 全场绘制验证以 ParentBuilt>=ParentTotal 为前提，增量模式永远不满足，
            // 且其呈现锁已在构造时解开——增量模式跳过该验证。
            if (!_incrementalMode)
                ValidateParentDrawSubmission();
        }

        private void ValidateParentDrawSubmission()
        {
            // Do not wait for every optional child16 rescue page.  Parent32 is
            // the resident mesh and must prove that it reached the renderer as
            // soon as all parent pages are committed.
            if (_drawValidationLogged || ParentBuilt < ParentTotal)
                return;

            if (_drawValidationEarliestFrame < 0)
            {
                _drawValidationEarliestFrame = Time.frameCount + 2;
                return;
            }
            if (Time.frameCount < _drawValidationEarliestFrame)
                return;

            long eligible = _parent32.VisibleKnownDrawVertexCount;
            long submitted = _parent32.RecentlySubmittedVertexCount;
            if (eligible > 0 && submitted == 0)
            {
                Logger.Error(
                    $"HERA parent draw validation failed: eligible_indices={eligible}, " +
                    "submitted_indices=0. Rescue overlays are hidden from validation; " +
                    "a rescue-only frame is not a valid completed result.");
            }
            else
            {
                Logger.Info(
                    $"HERA parent draw validation: eligible_indices={eligible}, " +
                    $"submitted_indices={submitted}, direct_snapshot_draw=1");
                _rescuePresentationUnlocked = true;
                _child16.SetVisible(_visible);
            }
            _drawValidationLogged = true;
        }

        public void SetVisible(bool visible)
        {
            _visible = visible;
            _parent32.SetVisible(visible);
            _child16.SetVisible(visible && _rescuePresentationUnlocked);
        }

        public void SetDiagnosticColoring(bool enabled)
        {
            _diagnosticColoring = enabled;
            _parent32.SetDiagnosticColoring(enabled);
            _child16.SetDiagnosticColoring(enabled);
        }

        // ── 增量精修公开 API（两段合一）──

        public bool IsIncremental => _incrementalMode;
        /// <summary>已提交的父页数（HUD"精修上屏 n 页"；重提交不重复计数）。</summary>
        public int IncrementalPagesCommitted => _parentResults;
        /// <summary>父+子管线提交看门狗复位总数（HUD 回读丢弃活度）。</summary>
        public int CommitWatchdogResets =>
            _parent32.CommitWatchdogResets + _child16.CommitWatchdogResets;
        /// <summary>父页排队深度（HUD 拥塞判读）。</summary>
        public int ParentQueueDepth => _disposed ? 0 : _parent32.PendingChunkCount;
        /// <summary>父页提交在途数（HUD 拥塞判读）。</summary>
        public int ParentCommitInFlight => _disposed ? 0 : _parent32.CommitPendingCount;
        /// <summary>父页入队→落地墙钟 EMA（ms，HUD 计时账）。</summary>
        public float ParentAvgQueueToCommitMs => _disposed ? 0f : _parent32.AvgQueueToCommitMs;
        /// <summary>父页派发→回读回调往返 EMA（ms，HUD 计时账）。</summary>
        public float ParentAvgDispatchToCallbackMs => _disposed ? 0f : _parent32.AvgDispatchToCallbackMs;

        /// <summary>
        /// 冻结成功的 32³ 块入场精修。父页网格与冻结网格同尺寸同原点（已核验同构：
        /// 体积 192×128×192 下双方都是 6×4×6，线性化公式一致），坐标直译。
        /// 邻页不主动刷新：冻结只翻 weight 符号不改 abs 几何，共享面体素的证据
        /// 在邻块冻结前本就未定格，刷了也是旧值——等邻块自己冻结时重排即可。
        /// </summary>
        public bool QueueParentBlock(int3 coordinate)
        {
            if (_disposed || !_incrementalMode) return false;
            // 定稿轨接管：移出实时集合，本次提交若有问题三角则照常建家族。
            _liveParentKeys.Remove(Flatten(coordinate, _parentGrid));
            if (!_parent32.QueueStaticReplayChunk(coordinate)) return false;
            // 解冻期被隐藏的页在重冻重建后要恢复可见（非家族页没人替它开）。
            _parent32.SetChunkVisible(coordinate, true);
            return true;
        }

        /// <summary>
        /// 实时轨：未冻块即时出粗网格页。与 QueueParentBlock 的唯一差别是提交后
        /// 不建家族（16³ 救回/边界补丁留给冻结后的定稿轨），tally/红占比照常记账，
        /// HUD 视线块状态不受影响。页随块级节流周期性重提交=缺肉长肉直播。
        /// </summary>
        public bool QueueLiveParentBlock(int3 coordinate)
        {
            if (_disposed || !_incrementalMode) return false;
            if (!_parent32.QueueStaticReplayChunk(coordinate)) return false;
            _liveParentKeys.Add(Flatten(coordinate, _parentGrid));
            _parent32.SetChunkVisible(coordinate, true);
            return true;
        }

        /// <summary>
        /// 块解冻：页面立刻撤下（旧内容已不可信），该页总账贡献同步扣回；
        /// 所属家族（若有）解散，子页影子补丁收起。重冻后 QueueParentBlock 重建，
        /// 家族随父页重提交自动重组、子页重新精修。
        /// </summary>
        public void InvalidateParentBlock(int3 coordinate)
        {
            if (_disposed || !_incrementalMode) return;
            _liveParentKeys.Remove(Flatten(coordinate, _parentGrid));
            _parent32.SetChunkVisible(coordinate, false);
            int parentKey = Flatten(coordinate, _parentGrid);
            if (_parentTally.TryGetValue(parentKey, out ParentTally tally))
            {
                _parentResults--;
                _parentKept -= tally.Kept;
                _parentDelegated -= tally.Delegated;
                _parentDisplayRed -= tally.DisplayRed;
                _parentConfirmationUnknown -= tally.ConfUnknown;
                _parentConfirmationPending -= tally.ConfPending;
                _parentConfirmationConfirmed -= tally.ConfConfirmed;
                _parentConfirmationMixed -= tally.ConfMixed;
                _parentInternalMixedContact -= tally.InternalContact;
                _parentInternalMixedRealEdge -= tally.InternalRealEdge;
                _parentInternalMixedSuspectedCenterFalse -= tally.InternalSuspected;
                _parentInternalMixedAmbiguous -= tally.InternalAmbiguous;
                _parentDelegatedInternalMixedRealEdge -= tally.DelegatedRealEdge;
                _parentDelegatedInternalMixedSuspectedCenterFalse -= tally.DelegatedSuspected;
                _parentDelegatedInternalMixedAmbiguous -= tally.DelegatedAmbiguous;
                _parentDelegatedTransition -= tally.DelegatedTransition;
                int3 ledgerCoordinate = coordinate / 2;
                int ledgerKey = Flatten(ledgerCoordinate, _ledgerGrid);
                if (_ledger64.TryGetValue(ledgerKey, out Ledger64Entry ledger))
                {
                    ledger.ParentPagesCompleted--;
                    ledger.ParentKeptTriangles -= tally.Kept;
                    ledger.ParentDelegatedTriangles -= tally.Delegated;
                    if (tally.Delegated > 0) ledger.ProblemParentPages--;
                }
                _parentTally.Remove(parentKey);
            }
            if (_families.TryGetValue(parentKey, out Family family))
                RemoveFamilyIncremental(parentKey, family);
        }

        private void RemoveFamilyIncremental(int parentKey, Family family)
        {
            long interiorShadow = 0, boundaryShadow = 0;
            for (int i = 0; i < family.Children.Count; i++)
            {
                int3 child = family.Children[i];
                // 子页本体从未可见（16³ 只救回不换页），只需收影子补丁。
                _child16.SetHeraInteriorShadowVisible(child, false);
                _child16.SetHeraBoundaryShadowVisible(child, false);
                _childToParent.Remove(Flatten(child, _childGrid));
            }
            for (int i = 0; i < family.TrianglePages.Count; i++)
            {
                interiorShadow += family.TrianglePages[i].InteriorShadowTriangles;
                boundaryShadow += family.TrianglePages[i].BoundaryShadowTriangles;
            }
            // 子页提交总账回退（重冻后家族重建会重新入账）。
            _childResults -= family.CompletedChildren;
            _childKept -= family.ChildKept;
            _childDelegated -= family.ChildDelegated;
            _triangleInteriorShadow -= interiorShadow;
            _triangleBoundaryShadow -= boundaryShadow;
            _familiesQueued--;
            if (_ledger64.TryGetValue(family.Ledger64Key, out Ledger64Entry ledger))
                ledger.FamiliesQueued--;
            if (family.Finalized)
            {
                _familiesBlocked--; // FinalizeFamily 恒走 blocked（整页替换已退役）
                if (family.MixedCommitted)
                {
                    _familiesMixedCommitted--;
                    _mixedCommittedPages -= family.MixedCommittedPages;
                    _mixedCommittedTriangles -= family.MixedCommittedTriangles;
                }
                _triangleBoundaryCommitted -= family.TriangleBoundaryCommitted;
                _boundaryCommittedPages -= family.BoundaryCommittedPages;
                if (_ledger64.TryGetValue(family.Ledger64Key, out Ledger64Entry l2))
                    l2.FamiliesBlocked--;
            }
            _families.Remove(parentKey);
        }

        /// <summary>体积清空时复位增量账（管道自身订阅 Cleared 复位子页，这里管家族/总账）。</summary>
        public void ResetIncrementalState()
        {
            if (_disposed || !_incrementalMode) return;
            _families.Clear();
            _childToParent.Clear();
            _ledger64.Clear();
            _parentTally.Clear();
            _liveParentKeys.Clear();
            _parentResults = 0;
            _childResults = 0;
            _familiesQueued = 0;
            _familiesSwapped = 0;
            _familiesBlocked = 0;
            _familiesMixedCommitted = 0;
            _mixedCommittedPages = 0;
            _mixedCommittedTriangles = 0;
            _boundaryCandidatePassed = 0;
            _boundaryCandidateBlocked = 0;
            _boundaryCandidateRecoverable = 0;
            _parentKept = 0; _parentDelegated = 0; _parentDisplayRed = 0;
            _parentConfirmationUnknown = 0; _parentConfirmationPending = 0;
            _parentConfirmationConfirmed = 0; _parentConfirmationMixed = 0;
            _childKept = 0; _childDelegated = 0;
            _parentInternalMixedContact = 0; _parentInternalMixedRealEdge = 0;
            _parentInternalMixedSuspectedCenterFalse = 0; _parentInternalMixedAmbiguous = 0;
            _parentDelegatedInternalMixedRealEdge = 0; _parentDelegatedInternalMixedSuspectedCenterFalse = 0;
            _parentDelegatedInternalMixedAmbiguous = 0; _parentDelegatedTransition = 0;
            _childInternalMixedContact = 0; _childInternalMixedRealEdge = 0;
            _childInternalMixedSuspectedCenterFalse = 0; _childInternalMixedAmbiguous = 0;
            _childDelegatedInternalMixedRealEdge = 0; _childDelegatedInternalMixedSuspectedCenterFalse = 0;
            _childDelegatedInternalMixedAmbiguous = 0; _childDelegatedTransition = 0;
            _triangleInteriorRecoverable = 0; _triangleInteriorShadow = 0;
            _trianglePureBoundaryCandidate = 0; _triangleBoundaryRecoverable = 0;
            _triangleBoundaryShadow = 0; _triangleBoundaryCommitted = 0;
            _triangleBoundaryBlockedExactMissing = 0; _triangleBoundaryShadowUnexpected = 0;
            _boundaryCommittedPages = 0;
            _triangleBoundaryBlockedCoverage = 0; _triangleBoundaryBlockedHard = 0;
            _triangleBoundaryBlockedMulti = 0; _triangleBoundaryBlockedTransition = 0;
            _triangleRiskRealEdge = 0; _triangleRiskNormal = 0; _triangleRiskWeight = 0;
            _triangleRiskMulti = 0; _triangleRiskTransition = 0;
        }

        /// <summary>HUD 视线块："精修中"判定（在队或提交在途）。</summary>
        public bool IsParentBlockInFlight(int3 coordinate)
        {
            return !_disposed && _parent32.IsChunkInFlight(coordinate);
        }

        /// <summary>HUD 视线块：已提交页的红三角数与总三角数（红=待证+混证，定稿前申诉期用）。</summary>
        public bool TryGetParentPageTally(int3 coordinate, out long redTriangles, out long totalTriangles)
        {
            redTriangles = 0;
            totalTriangles = 0;
            if (_disposed) return false;
            if (!_parentTally.TryGetValue(Flatten(coordinate, _parentGrid), out ParentTally tally))
                return false;
            redTriangles = tally.DisplayRed;
            totalTriangles = tally.Kept + tally.Delegated;
            return true;
        }

        private void OnParentCommitted(PersistentChunkMeshPipeline.StaticReplayPageResult result)
        {
            if (_disposed) return;
            int parentKey = Flatten(result.Coordinate, _parentGrid);
            // 增量模式下同一页可能重提交（解冻重修）：先扣该页旧账再入新账，
            // 全局总账不随重建次数膨胀。全场回放每页只提交一次，tally 恒为空，
            // 行为与原路径逐字节一致。
            _parentTally.TryGetValue(parentKey, out ParentTally old);
            long oldKept = old != null ? old.Kept : 0;
            long oldDelegated = old != null ? old.Delegated : 0;
            if (old == null) _parentResults++;
            _parentKept += result.KeptTriangles - oldKept;
            _parentDelegated += result.DelegatedTriangles - oldDelegated;
            _parentDisplayRed += result.DisplayRedTriangles - (old != null ? old.DisplayRed : 0);
            _parentConfirmationUnknown += result.ConfirmationUnknownTriangles - (old != null ? old.ConfUnknown : 0);
            _parentConfirmationPending += result.ConfirmationPendingTriangles - (old != null ? old.ConfPending : 0);
            _parentConfirmationConfirmed += result.ConfirmationConfirmedTriangles - (old != null ? old.ConfConfirmed : 0);
            _parentConfirmationMixed += result.ConfirmationMixedTriangles - (old != null ? old.ConfMixed : 0);
            _parentInternalMixedContact += result.InternalMixedContactTriangles - (old != null ? old.InternalContact : 0);
            _parentInternalMixedRealEdge += result.InternalMixedRealEdgeTriangles - (old != null ? old.InternalRealEdge : 0);
            _parentInternalMixedSuspectedCenterFalse += result.InternalMixedSuspectedCenterFalseTriangles - (old != null ? old.InternalSuspected : 0);
            _parentInternalMixedAmbiguous += result.InternalMixedAmbiguousTriangles - (old != null ? old.InternalAmbiguous : 0);
            _parentDelegatedInternalMixedRealEdge += result.DelegatedInternalMixedRealEdgeTriangles - (old != null ? old.DelegatedRealEdge : 0);
            _parentDelegatedInternalMixedSuspectedCenterFalse += result.DelegatedInternalMixedSuspectedCenterFalseTriangles - (old != null ? old.DelegatedSuspected : 0);
            _parentDelegatedInternalMixedAmbiguous += result.DelegatedInternalMixedAmbiguousTriangles - (old != null ? old.DelegatedAmbiguous : 0);
            _parentDelegatedTransition += result.DelegatedTransitionTriangles - (old != null ? old.DelegatedTransition : 0);
            _parentTally[parentKey] = new ParentTally
            {
                Kept = result.KeptTriangles,
                Delegated = result.DelegatedTriangles,
                DisplayRed = result.DisplayRedTriangles,
                ConfUnknown = result.ConfirmationUnknownTriangles,
                ConfPending = result.ConfirmationPendingTriangles,
                ConfConfirmed = result.ConfirmationConfirmedTriangles,
                ConfMixed = result.ConfirmationMixedTriangles,
                InternalContact = result.InternalMixedContactTriangles,
                InternalRealEdge = result.InternalMixedRealEdgeTriangles,
                InternalSuspected = result.InternalMixedSuspectedCenterFalseTriangles,
                InternalAmbiguous = result.InternalMixedAmbiguousTriangles,
                DelegatedRealEdge = result.DelegatedInternalMixedRealEdgeTriangles,
                DelegatedSuspected = result.DelegatedInternalMixedSuspectedCenterFalseTriangles,
                DelegatedAmbiguous = result.DelegatedInternalMixedAmbiguousTriangles,
                DelegatedTransition = result.DelegatedTransitionTriangles
            };
            int3 ledgerCoordinate = result.Coordinate / 2;
            int ledgerKey = Flatten(ledgerCoordinate, _ledgerGrid);
            if (!_ledger64.TryGetValue(ledgerKey, out Ledger64Entry ledger))
            {
                ledger = new Ledger64Entry { Coordinate = ledgerCoordinate };
                _ledger64.Add(ledgerKey, ledger);
            }
            if (old == null) ledger.ParentPagesCompleted++;
            ledger.ParentKeptTriangles += result.KeptTriangles - oldKept;
            ledger.ParentDelegatedTriangles += result.DelegatedTriangles - oldDelegated;

            if (result.DelegatedTriangles > 0)
            {
                if (old == null || oldDelegated <= 0) ledger.ProblemParentPages++;
            }
            else if (oldDelegated > 0)
                ledger.ProblemParentPages--;

            if (result.DelegatedTriangles <= 0)
                return;
            // 实时轨页只出粗网格：不建家族不派生子页，tally 已在上面记完。
            if (_liveParentKeys.Contains(parentKey))
                return;
            // 重提交不重建家族：边界刷新型重建不改变家族裁决；解冻重修型重建
            // 已在 InvalidateParentBlock 里先移除家族，走到这里必是新家族。
            if (_families.ContainsKey(parentKey))
                return;

            var family = new Family
            {
                Parent = result.Coordinate,
                Ledger64Key = ledgerKey,
                ParentKept = result.KeptTriangles,
                ParentDelegated = result.DelegatedTriangles,
                ParentAmbiguousNormal = result.DelegatedAmbiguousNormalTriangles,
                ParentAmbiguousWeight = result.DelegatedAmbiguousWeightTriangles,
                ParentAmbiguousBoundary = result.DelegatedAmbiguousBoundaryTriangles,
                ParentAmbiguousMulti = result.DelegatedAmbiguousMultiFailureTriangles,
                ParentRealEdge = result.DelegatedInternalMixedRealEdgeTriangles,
                ParentSuspectedCenterFalse = result.DelegatedInternalMixedSuspectedCenterFalseTriangles,
                ParentTransition = result.DelegatedTransitionTriangles
            };

            int3 childBase = result.Coordinate * 2;
            for (int z = 0; z < 2; z++)
            for (int y = 0; y < 2; y++)
            for (int x = 0; x < 2; x++)
            {
                int3 child = childBase + new int3(x, y, z);
                if (math.any(child < 0) || math.any(child >= _childGrid))
                    continue;
                int childKey = Flatten(child, _childGrid);
                _child16.SetChunkVisible(child, false);
                _child16.SetHeraInteriorShadowVisible(child, false);
                _child16.SetHeraBoundaryShadowVisible(child, false);
                if (_child16.QueueStaticReplayChunk(child))
                {
                    family.Children.Add(child);
                    _childToParent[childKey] = parentKey;
                    family.ExpectedChildren++;
                }
            }

            _families[parentKey] = family;
            _familiesQueued++;
            ledger.FamiliesQueued++;
            if (family.ExpectedChildren == 0)
                FinalizeFamily(family);
        }

        private void OnChildCommitted(PersistentChunkMeshPipeline.StaticReplayPageResult result)
        {
            if (_disposed) return;
            int childKey = Flatten(result.Coordinate, _childGrid);
            if (!_childToParent.TryGetValue(childKey, out int parentKey) ||
                !_families.TryGetValue(parentKey, out Family family) || family.Finalized)
                return;

            _childResults++;
            _childKept += result.KeptTriangles;
            _childDelegated += result.DelegatedTriangles;
            _childInternalMixedContact += result.InternalMixedContactTriangles;
            _childInternalMixedRealEdge += result.InternalMixedRealEdgeTriangles;
            _childInternalMixedSuspectedCenterFalse += result.InternalMixedSuspectedCenterFalseTriangles;
            _childInternalMixedAmbiguous += result.InternalMixedAmbiguousTriangles;
            _childDelegatedInternalMixedRealEdge += result.DelegatedInternalMixedRealEdgeTriangles;
            _childDelegatedInternalMixedSuspectedCenterFalse += result.DelegatedInternalMixedSuspectedCenterFalseTriangles;
            _childDelegatedInternalMixedAmbiguous += result.DelegatedInternalMixedAmbiguousTriangles;
            _childDelegatedTransition += result.DelegatedTransitionTriangles;
            _triangleInteriorShadow += result.InteriorShadowTriangles;
            _triangleBoundaryShadow += result.BoundaryShadowTriangles;
            family.CompletedChildren++;
            family.ChildKept += result.KeptTriangles;
            family.ChildDelegated += result.DelegatedTriangles;
            family.ChildAmbiguousNormal += result.DelegatedAmbiguousNormalTriangles;
            family.ChildAmbiguousWeight += result.DelegatedAmbiguousWeightTriangles;
            family.ChildAmbiguousBoundary += result.DelegatedAmbiguousBoundaryTriangles;
            family.ChildAmbiguousMulti += result.DelegatedAmbiguousMultiFailureTriangles;
            family.ChildRealEdge += result.DelegatedInternalMixedRealEdgeTriangles;
            family.ChildSuspectedCenterFalse += result.DelegatedInternalMixedSuspectedCenterFalseTriangles;
            family.ChildTransition += result.DelegatedTransitionTriangles;
            family.TrianglePages.Add(new TriangleCounterfactualPage
            {
                Coordinate = result.Coordinate,
                SuspectedCenterFalse = result.DelegatedInternalMixedSuspectedCenterFalseTriangles,
                InteriorShadowTriangles = result.InteriorShadowTriangles,
                BoundaryShadowTriangles = result.BoundaryShadowTriangles,
                PureBoundary = result.DelegatedAmbiguousBoundaryTriangles,
                RealEdge = result.DelegatedInternalMixedRealEdgeTriangles,
                AmbiguousNormal = result.DelegatedAmbiguousNormalTriangles,
                AmbiguousWeight = result.DelegatedAmbiguousWeightTriangles,
                AmbiguousMulti = result.DelegatedAmbiguousMultiFailureTriangles,
                Transition = result.DelegatedTransitionTriangles
            });
            // Do not expose a half-complete family.  Exact local patches become
            // visible only after FinalizeFamily has evaluated the complete
            // parent/child ledger.
            _child16.SetHeraInteriorShadowVisible(result.Coordinate, false);
            _child16.SetHeraBoundaryShadowVisible(result.Coordinate, false);
            if (family.CompletedChildren >= family.ExpectedChildren)
                FinalizeFamily(family);
        }

        private void FinalizeFamily(Family family)
        {
            if (family.Finalized) return;
            family.Finalized = true;

            family.AcceptedCoverageDelta = family.ChildKept - family.ParentKept;
            family.DelegatedReduction = family.ParentDelegated - family.ChildDelegated;
            family.BoundaryOnlyReduction =
                family.ParentAmbiguousBoundary - family.ChildAmbiguousBoundary;
            family.HardConflictDelta =
                (family.ChildRealEdge + family.ChildAmbiguousNormal + family.ChildAmbiguousWeight) -
                (family.ParentRealEdge + family.ParentAmbiguousNormal + family.ParentAmbiguousWeight);
            family.TransitionDelta = family.ChildTransition - family.ParentTransition;
            family.SuspectedCenterFalseGain =
                family.ChildSuspectedCenterFalse - family.ParentSuspectedCenterFalse;

            // Exact triangle admission.  The GPU boundary shadow stream is
            // already formed one triangle at a time and contains only the
            // canonical, fine-tier, pure-boundary case: normal, weight and
            // multi-failure triangles never enter it.  Therefore an unrelated
            // family-wide regression must not veto every safe triangle in the
            // page.  Family tests below remain as a legacy counterfactual only.
            bool triangleCoverageSafe = family.AcceptedCoverageDelta >= 0;
            bool triangleHardSafe = family.HardConflictDelta <= 0;
            bool triangleMultiSafe = family.ChildAmbiguousMulti <= family.ParentAmbiguousMulti;
            bool triangleTransitionSafe = family.TransitionDelta <= 0;
            for (int i = 0; i < family.TrianglePages.Count; i++)
            {
                TriangleCounterfactualPage page = family.TrianglePages[i];
                page.InteriorRecoverable = page.SuspectedCenterFalse;
                family.TriangleInteriorRecoverable += page.InteriorRecoverable;
                family.TriangleInteriorShadow += page.InteriorShadowTriangles;
                family.TriangleBoundaryShadow += page.BoundaryShadowTriangles;
                family.TrianglePureBoundaryCandidate += page.PureBoundary;

                page.BoundaryRecoverable =
                    Math.Min(page.PureBoundary, page.BoundaryShadowTriangles);
                page.BoundaryBlockedExactMissing =
                    Math.Max(0, page.PureBoundary - page.BoundaryShadowTriangles);
                page.BoundaryShadowUnexpected =
                    Math.Max(0, page.BoundaryShadowTriangles - page.PureBoundary);

                // Preserve the old family-wide verdict as a read-only control
                // ledger so the effect of removing collective punishment stays
                // directly measurable against the frozen baseline.
                if (page.PureBoundary > 0)
                {
                    if (!triangleCoverageSafe)
                        page.BoundaryBlockedCoverage = page.PureBoundary;
                    else if (!triangleHardSafe)
                        page.BoundaryBlockedHard = page.PureBoundary;
                    else if (!triangleMultiSafe)
                        page.BoundaryBlockedMulti = page.PureBoundary;
                    else if (!triangleTransitionSafe)
                        page.BoundaryBlockedTransition = page.PureBoundary;
                }

                family.TriangleBoundaryRecoverable += page.BoundaryRecoverable;
                family.TriangleBoundaryBlockedExactMissing += page.BoundaryBlockedExactMissing;
                family.TriangleBoundaryShadowUnexpected += page.BoundaryShadowUnexpected;
                family.TriangleBoundaryBlockedCoverage += page.BoundaryBlockedCoverage;
                family.TriangleBoundaryBlockedHard += page.BoundaryBlockedHard;
                family.TriangleBoundaryBlockedMulti += page.BoundaryBlockedMulti;
                family.TriangleBoundaryBlockedTransition += page.BoundaryBlockedTransition;
                _triangleRiskRealEdge += page.RealEdge;
                _triangleRiskNormal += page.AmbiguousNormal;
                _triangleRiskWeight += page.AmbiguousWeight;
                _triangleRiskMulti += page.AmbiguousMulti;
                _triangleRiskTransition += page.Transition;
            }
            _triangleInteriorRecoverable += family.TriangleInteriorRecoverable;
            _trianglePureBoundaryCandidate += family.TrianglePureBoundaryCandidate;
            _triangleBoundaryRecoverable += family.TriangleBoundaryRecoverable;
            _triangleBoundaryBlockedExactMissing += family.TriangleBoundaryBlockedExactMissing;
            _triangleBoundaryShadowUnexpected += family.TriangleBoundaryShadowUnexpected;
            _triangleBoundaryBlockedCoverage += family.TriangleBoundaryBlockedCoverage;
            _triangleBoundaryBlockedHard += family.TriangleBoundaryBlockedHard;
            _triangleBoundaryBlockedMulti += family.TriangleBoundaryBlockedMulti;
            _triangleBoundaryBlockedTransition += family.TriangleBoundaryBlockedTransition;

            // Read-only boundary-aware candidate gate.  A page-boundary label is
            // treated as an ownership/halo symptom, not automatically as a
            // geometric failure.  This counterfactual deliberately does not
            // change visibility or the production atomic replacement decision.
            bool candidatePreservesCoverage = family.AcceptedCoverageDelta >= 0;
            bool candidateDoesNotIncreaseDelegation = family.DelegatedReduction >= 0;
            bool candidateRelievesPureBoundary = family.BoundaryOnlyReduction > 0;
            bool candidateDoesNotIncreaseHardConflict = family.HardConflictDelta <= 0;
            bool candidateDoesNotIncreaseMultiFailure =
                family.ChildAmbiguousMulti <= family.ParentAmbiguousMulti;
            bool candidateDoesNotWorsenTransition = family.TransitionDelta <= 0;
            family.BoundaryCandidatePass =
                candidatePreservesCoverage &&
                candidateDoesNotIncreaseDelegation &&
                candidateRelievesPureBoundary &&
                candidateDoesNotIncreaseHardConflict &&
                candidateDoesNotIncreaseMultiFailure &&
                candidateDoesNotWorsenTransition;
            if (family.BoundaryCandidatePass)
            {
                family.BoundaryCandidateReason = "boundary_relief_without_semantic_or_seam_regression";
                _boundaryCandidatePassed++;
                _boundaryCandidateRecoverable += family.BoundaryOnlyReduction;
            }
            else
            {
                family.BoundaryCandidateReason =
                    !candidatePreservesCoverage ? "candidate_coverage_loss" :
                    !candidateDoesNotIncreaseDelegation ? "candidate_delegation_increase" :
                    !candidateRelievesPureBoundary ? "candidate_no_pure_boundary_relief" :
                    !candidateDoesNotIncreaseHardConflict ? "candidate_hard_conflict_increase" :
                    !candidateDoesNotIncreaseMultiFailure ? "candidate_multi_failure_increase" :
                    "candidate_transition_increase";
                _boundaryCandidateBlocked++;
            }

            // Keep the former full-page hand-off calculation as a read-only
            // counterfactual. child16 is a rescue source only: it may add exact
            // triangles that parent32 could not decide, but it can never hide or
            // replace the resident parent page.
            bool preservesAcceptedCoverage = family.AcceptedCoverageDelta >= 0;
            bool reducesDelegation = family.DelegatedReduction > 0;
            bool legacyFullPagePass = preservesAcceptedCoverage && reducesDelegation;
            family.DecisionReason = legacyFullPagePass
                ? "child16_rescue_only_legacy_full_page_would_pass"
                : "child16_rescue_only_legacy_full_page_blocked";

            _parent32.SetChunkVisible(family.Parent, true);
            for (int i = 0; i < family.Children.Count; i++)
            {
                _child16.SetChunkVisible(family.Children[i], false);
                _child16.SetHeraInteriorShadowVisible(family.Children[i], false);
                _child16.SetHeraBoundaryShadowVisible(family.Children[i], false);
            }

            // Retain the complete parent page and add only exact child16
            // triangles. Real edges, ambiguous normals/weights and transitions
            // never gain page-level authority.
            for (int i = 0; i < family.TrianglePages.Count; i++)
            {
                TriangleCounterfactualPage page = family.TrianglePages[i];
                bool acceptLocalPatch = page.InteriorShadowTriangles > 0;
                bool acceptBoundaryPatch =
                    page.BoundaryRecoverable > 0 &&
                    page.BoundaryShadowTriangles == page.BoundaryRecoverable;
                _child16.SetHeraInteriorShadowVisible(page.Coordinate, acceptLocalPatch);
                _child16.SetHeraBoundaryShadowVisible(page.Coordinate, acceptBoundaryPatch);
                if (acceptBoundaryPatch)
                {
                    page.BoundaryCommittedTriangles = page.BoundaryRecoverable;
                    family.TriangleBoundaryCommitted += page.BoundaryCommittedTriangles;
                    family.BoundaryCommittedPages++;
                }
                if (!acceptLocalPatch && !acceptBoundaryPatch) continue;
                family.MixedCommittedPages++;
                if (acceptLocalPatch)
                    family.MixedCommittedTriangles += page.InteriorShadowTriangles;
                if (acceptBoundaryPatch)
                    family.MixedCommittedTriangles += page.BoundaryCommittedTriangles;
            }
            if (family.MixedCommittedTriangles > 0)
            {
                family.MixedCommitted = true;
                _familiesMixedCommitted++;
                _mixedCommittedPages += family.MixedCommittedPages;
                _mixedCommittedTriangles += family.MixedCommittedTriangles;
            }

            // Full child-page replacement is deliberately disabled. Count it
            // as blocked so old reports cannot mistake rescue-only operation
            // for a missing family decision.
            family.Swapped = false;
            _familiesBlocked++;
            if (_ledger64.TryGetValue(family.Ledger64Key, out Ledger64Entry ledger))
            {
                ledger.FamiliesBlocked++;
                ledger.Version++;
            }
            _triangleBoundaryCommitted += family.TriangleBoundaryCommitted;
            _boundaryCommittedPages += family.BoundaryCommittedPages;
        }

        public void AppendReport(StringBuilder sb)
        {
            sb.AppendLine("ScanCover HERA frozen TSDF replay");
            sb.AppendLine($"utc={DateTime.UtcNow:O}");
            sb.AppendLine($"ledger_complete={IsComplete.ToString().ToLowerInvariant()}");
            sb.AppendLine($"replay_failed={Failed.ToString().ToLowerInvariant()}");
            sb.AppendLine($"families_finalized={FamiliesFinalized}/{FamiliesQueued}");
            sb.AppendLine($"families_pending={FamiliesPending}");
            sb.AppendLine($"children_pending={ChildrenPending}");
            sb.AppendLine("roles=64_lifetime_ledger,32_evidence_router,16_on_demand_refinement");
            sb.AppendLine("replacement=parent32_resident_plus_child16_exact_rescue_only");
            sb.AppendLine("child16_full_page_replacement_enabled=false");
            sb.AppendLine("child16_can_remove_parent32=false");
            sb.AppendLine("production_overwrite=false");
            sb.AppendLine("decision_state_inheritance=false; classification_semantics_inheritance=true");
            sb.AppendLine("same_family_delta_scope=delegated_ambiguous_reason_counts; exact_triangle_ids_are_not_cross_resolution_comparable");
            sb.AppendLine("boundary_admission_granularity=exact_triangle_gpu");
            sb.AppendLine("boundary_family_gate=read_only_legacy_counterfactual");
            sb.AppendLine("boundary_candidate_gate_is_read_only=true");
            sb.AppendLine("boundary_candidate_semantics=pure_page_boundary_is_ownership_or_halo_evidence; real_edge,normal,weight,multi_failure,and_transition_remain_safety_guardrails");
            sb.AppendLine("triangle_counterfactual_is_read_only=false_for_exact_interior_and_boundary_streams");
            sb.AppendLine("triangle_counterfactual_semantics=child16 per-triangle buckets; interior suspected-center-false is locally recoverable; pure-boundary requires family coverage,hard-conflict,multi-failure,and-transition guardrails");
            sb.AppendLine("triangle_counterfactual_does_not_compare_cross_resolution_triangle_ids=true");
            sb.AppendLine("interior_shadow_exact_triangle_stream=true");
            sb.AppendLine("interior_shadow_display=green_exact_local_patch");
            sb.AppendLine("boundary_shadow_exact_triangle_stream=true");
            sb.AppendLine("boundary_shadow_owner=generateindices_half_open_core_source_page");
            sb.AppendLine("boundary_shadow_display=green_exact_local_patch");
            sb.AppendLine("triangle_identity_palette=disabled");
            sb.AppendLine("triangle_index_layout=bits_0_29_vertex_id,bits_30_31_route");
            sb.AppendLine("triangle_color_legend=green:receiver_confirmed_resident_or_child16_rescue,red:unknown_pending_or_mixed_confirmation");
            sb.AppendLine("display=dual_route_always_visible");
            sb.AppendLine($"ledger64={Ledger64Count}/{Ledger64Total}");
            sb.AppendLine($"parent32_results={_parentResults}/{ParentTotal}");
            sb.AppendLine($"parent32_kept_triangles={_parentKept}");
            sb.AppendLine($"parent32_delegated_triangles={_parentDelegated}");
            sb.AppendLine($"parent32_internal_mixed_real_edge_triangles={_parentInternalMixedRealEdge}");
            sb.AppendLine($"parent32_internal_mixed_suspected_center_false_triangles={_parentInternalMixedSuspectedCenterFalse}");
            sb.AppendLine($"parent32_internal_mixed_ambiguous_triangles={_parentInternalMixedAmbiguous}");
            long parentContactClassified = _parentInternalMixedRealEdge +
                                      _parentInternalMixedSuspectedCenterFalse +
                                      _parentInternalMixedAmbiguous;
            long parentDelegatedClassified = _parentDelegatedInternalMixedRealEdge +
                                             _parentDelegatedInternalMixedSuspectedCenterFalse +
                                             _parentDelegatedInternalMixedAmbiguous;
            sb.AppendLine($"parent32_internal_mixed_contact_triangles={_parentInternalMixedContact}");
            sb.AppendLine($"parent32_internal_mixed_contact_classified_total={parentContactClassified}");
            sb.AppendLine($"parent32_internal_mixed_contact_reconcile_delta={_parentInternalMixedContact - parentContactClassified}");
            sb.AppendLine($"parent32_delegated_internal_mixed_real_edge_triangles={_parentDelegatedInternalMixedRealEdge}");
            sb.AppendLine($"parent32_delegated_internal_mixed_suspected_center_false_triangles={_parentDelegatedInternalMixedSuspectedCenterFalse}");
            sb.AppendLine($"parent32_delegated_internal_mixed_ambiguous_triangles={_parentDelegatedInternalMixedAmbiguous}");
            sb.AppendLine($"parent32_delegated_internal_mixed_classified_total={parentDelegatedClassified}");
            sb.AppendLine($"parent32_delegated_transition_triangles={_parentDelegatedTransition}");
            sb.AppendLine($"parent32_delegated_ledger_reconcile_delta={_parentDelegated - parentDelegatedClassified - _parentDelegatedTransition}");
            sb.AppendLine($"child16_results={_childResults}/{ChildQueued}");
            sb.AppendLine($"child16_kept_triangles={_childKept}");
            sb.AppendLine($"child16_delegated_triangles={_childDelegated}");
            sb.AppendLine($"child16_internal_mixed_real_edge_triangles={_childInternalMixedRealEdge}");
            sb.AppendLine($"child16_internal_mixed_suspected_center_false_triangles={_childInternalMixedSuspectedCenterFalse}");
            sb.AppendLine($"child16_internal_mixed_ambiguous_triangles={_childInternalMixedAmbiguous}");
            long childContactClassified = _childInternalMixedRealEdge +
                                     _childInternalMixedSuspectedCenterFalse +
                                     _childInternalMixedAmbiguous;
            long childDelegatedClassified = _childDelegatedInternalMixedRealEdge +
                                            _childDelegatedInternalMixedSuspectedCenterFalse +
                                            _childDelegatedInternalMixedAmbiguous;
            sb.AppendLine($"child16_internal_mixed_contact_triangles={_childInternalMixedContact}");
            sb.AppendLine($"child16_internal_mixed_contact_classified_total={childContactClassified}");
            sb.AppendLine($"child16_internal_mixed_contact_reconcile_delta={_childInternalMixedContact - childContactClassified}");
            sb.AppendLine($"child16_delegated_internal_mixed_real_edge_triangles={_childDelegatedInternalMixedRealEdge}");
            sb.AppendLine($"child16_delegated_internal_mixed_suspected_center_false_triangles={_childDelegatedInternalMixedSuspectedCenterFalse}");
            sb.AppendLine($"child16_delegated_internal_mixed_ambiguous_triangles={_childDelegatedInternalMixedAmbiguous}");
            sb.AppendLine($"child16_delegated_internal_mixed_classified_total={childDelegatedClassified}");
            sb.AppendLine($"child16_delegated_transition_triangles={_childDelegatedTransition}");
            sb.AppendLine($"child16_delegated_ledger_reconcile_delta={_childDelegated - childDelegatedClassified - _childDelegatedTransition}");
            sb.AppendLine($"families_queued={_familiesQueued}");
            sb.AppendLine($"families_swapped={_familiesSwapped}");
            sb.AppendLine($"families_blocked={_familiesBlocked}");
            sb.AppendLine($"families_mixed_committed={_familiesMixedCommitted}");
            sb.AppendLine($"mixed_commit_pages={_mixedCommittedPages}");
            sb.AppendLine($"mixed_commit_exact_triangles={_mixedCommittedTriangles}");
            sb.AppendLine($"boundary_candidate_passed={_boundaryCandidatePassed}");
            sb.AppendLine($"boundary_candidate_blocked={_boundaryCandidateBlocked}");
            sb.AppendLine($"boundary_candidate_recoverable_boundary_triangles={_boundaryCandidateRecoverable}");
            sb.AppendLine($"triangle_cf_interior_false_recoverable={_triangleInteriorRecoverable}");
            sb.AppendLine($"triangle_shadow_interior_exact={_triangleInteriorShadow}");
            sb.AppendLine($"triangle_shadow_missing_from_expected={Math.Max(0L, _triangleInteriorRecoverable - _triangleInteriorShadow)}");
            sb.AppendLine($"triangle_cf_pure_boundary_candidate={_trianglePureBoundaryCandidate}");
            sb.AppendLine($"triangle_boundary_exact_recoverable={_triangleBoundaryRecoverable}");
            sb.AppendLine($"triangle_shadow_boundary_exact={_triangleBoundaryShadow}");
            sb.AppendLine($"triangle_boundary_exact_missing={_triangleBoundaryBlockedExactMissing}");
            sb.AppendLine($"triangle_boundary_exact_unexpected={_triangleBoundaryShadowUnexpected}");
            sb.AppendLine($"triangle_boundary_exact_committed={_triangleBoundaryCommitted}");
            sb.AppendLine($"triangle_boundary_exact_committed_pages={_boundaryCommittedPages}");
            sb.AppendLine($"triangle_boundary_exact_commit_blocker={(_triangleBoundaryCommitted == _triangleBoundaryRecoverable ? "none" : "unexpected_stream_or_commit_not_finalized")}");
            sb.AppendLine($"triangle_cf_total_recoverable={_triangleInteriorRecoverable + _triangleBoundaryRecoverable}");
            sb.AppendLine($"legacy_family_would_block_boundary_coverage={_triangleBoundaryBlockedCoverage}");
            sb.AppendLine($"legacy_family_would_block_boundary_hard={_triangleBoundaryBlockedHard}");
            sb.AppendLine($"legacy_family_would_block_boundary_multi={_triangleBoundaryBlockedMulti}");
            sb.AppendLine($"legacy_family_would_block_boundary_transition={_triangleBoundaryBlockedTransition}");
            sb.AppendLine($"triangle_cf_coexisting_real_edge={_triangleRiskRealEdge}");
            sb.AppendLine($"triangle_cf_coexisting_normal_failure={_triangleRiskNormal}");
            sb.AppendLine($"triangle_cf_coexisting_weight_failure={_triangleRiskWeight}");
            sb.AppendLine($"triangle_cf_coexisting_multi_failure={_triangleRiskMulti}");
            sb.AppendLine($"triangle_cf_coexisting_transition={_triangleRiskTransition}");
            long dualRed = _parentDisplayRed;
            long dualGreenParent = Math.Max(0L, _parentKept - _parentDisplayRed);
            long dualGreenRescue = _mixedCommittedTriangles;
            sb.AppendLine($"dual_color_parent32_green_triangles={dualGreenParent}");
            sb.AppendLine($"dual_color_parent32_red_triangles={dualRed}");
            sb.AppendLine($"dual_color_parent32_unknown_confirmation_triangles={_parentConfirmationUnknown}");
            sb.AppendLine($"dual_color_parent32_pending_confirmation_triangles={_parentConfirmationPending}");
            sb.AppendLine($"dual_color_parent32_confirmed_triangles={_parentConfirmationConfirmed}");
            sb.AppendLine($"dual_color_parent32_mixed_confirmation_triangles={_parentConfirmationMixed}");
            sb.AppendLine($"dual_color_child16_rescue_green_triangles={dualGreenRescue}");
            sb.AppendLine($"dual_color_visible_total={dualGreenParent + dualRed + dualGreenRescue}");
            sb.AppendLine($"dual_color_visible_reconcile_delta={VisibleTriangles - dualGreenParent - dualRed - dualGreenRescue}");
            sb.AppendLine($"stored_vertex_payload={StoredVertexPayload.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"visible_triangles={VisibleTriangles.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"parent32_direct_draw_eligible_indices={_parent32.VisibleKnownDrawVertexCount.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"parent32_direct_draw_recent_submitted_indices={_parent32.RecentlySubmittedVertexCount.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine();
            sb.AppendLine("ledger64_csv:");
            sb.AppendLine("ledger_x,ledger_y,ledger_z,version,parent_pages_completed,problem_parent_pages,parent_kept_triangles,parent_delegated_triangles,families_queued,families_swapped,families_blocked");
            foreach (Ledger64Entry ledger in _ledger64.Values)
            {
                sb.Append(ledger.Coordinate.x).Append(',')
                  .Append(ledger.Coordinate.y).Append(',')
                  .Append(ledger.Coordinate.z).Append(',')
                  .Append(ledger.Version).Append(',')
                  .Append(ledger.ParentPagesCompleted).Append(',')
                  .Append(ledger.ProblemParentPages).Append(',')
                  .Append(ledger.ParentKeptTriangles).Append(',')
                  .Append(ledger.ParentDelegatedTriangles).Append(',')
                  .Append(ledger.FamiliesQueued).Append(',')
                  .Append(ledger.FamiliesSwapped).Append(',')
                  .Append(ledger.FamiliesBlocked).AppendLine();
            }
            sb.AppendLine();
            sb.AppendLine("family32_to_16_csv:");
            sb.AppendLine("parent_x,parent_y,parent_z,parent_kept,parent_delegated,children_expected,children_completed,child_kept,child_delegated,accepted_coverage_delta,delegated_reduction,parent_real_edge,parent_suspected_center_false,parent_transition,parent_amb_normal,parent_amb_weight,parent_amb_boundary,parent_amb_multi,child_real_edge,child_suspected_center_false,child_transition,child_amb_normal,child_amb_weight,child_amb_boundary,child_amb_multi,delta_real_edge,delta_suspected_center_false,delta_transition,delta_amb_normal,delta_amb_weight,delta_amb_boundary,delta_amb_multi,boundary_only_reduction,hard_conflict_delta,boundary_candidate_pass,boundary_candidate_reason,triangle_interior_recoverable,triangle_interior_shadow,triangle_boundary_candidate,triangle_boundary_exact_recoverable,triangle_boundary_shadow,triangle_boundary_exact_missing,triangle_boundary_shadow_unexpected,triangle_boundary_committed,boundary_committed_pages,legacy_would_block_coverage,legacy_would_block_hard,legacy_would_block_multi,legacy_would_block_transition,production_decision_reason,finalized,swapped,mixed_committed,mixed_committed_pages,mixed_committed_triangles");
            foreach (Family family in _families.Values)
            {
                sb.Append(family.Parent.x).Append(',')
                  .Append(family.Parent.y).Append(',')
                  .Append(family.Parent.z).Append(',')
                  .Append(family.ParentKept).Append(',')
                  .Append(family.ParentDelegated).Append(',')
                  .Append(family.ExpectedChildren).Append(',')
                  .Append(family.CompletedChildren).Append(',')
                  .Append(family.ChildKept).Append(',')
                  .Append(family.ChildDelegated).Append(',')
                  .Append(family.AcceptedCoverageDelta).Append(',')
                  .Append(family.DelegatedReduction).Append(',')
                  .Append(family.ParentRealEdge).Append(',')
                  .Append(family.ParentSuspectedCenterFalse).Append(',')
                  .Append(family.ParentTransition).Append(',')
                  .Append(family.ParentAmbiguousNormal).Append(',')
                  .Append(family.ParentAmbiguousWeight).Append(',')
                  .Append(family.ParentAmbiguousBoundary).Append(',')
                  .Append(family.ParentAmbiguousMulti).Append(',')
                  .Append(family.ChildRealEdge).Append(',')
                  .Append(family.ChildSuspectedCenterFalse).Append(',')
                  .Append(family.ChildTransition).Append(',')
                  .Append(family.ChildAmbiguousNormal).Append(',')
                  .Append(family.ChildAmbiguousWeight).Append(',')
                  .Append(family.ChildAmbiguousBoundary).Append(',')
                  .Append(family.ChildAmbiguousMulti).Append(',')
                  .Append(family.ChildRealEdge - family.ParentRealEdge).Append(',')
                  .Append(family.SuspectedCenterFalseGain).Append(',')
                  .Append(family.TransitionDelta).Append(',')
                  .Append(family.ChildAmbiguousNormal - family.ParentAmbiguousNormal).Append(',')
                  .Append(family.ChildAmbiguousWeight - family.ParentAmbiguousWeight).Append(',')
                  .Append(family.ChildAmbiguousBoundary - family.ParentAmbiguousBoundary).Append(',')
                  .Append(family.ChildAmbiguousMulti - family.ParentAmbiguousMulti).Append(',')
                  .Append(family.BoundaryOnlyReduction).Append(',')
                  .Append(family.HardConflictDelta).Append(',')
                  .Append(family.BoundaryCandidatePass ? 1 : 0).Append(',')
                  .Append(family.BoundaryCandidateReason).Append(',')
                  .Append(family.TriangleInteriorRecoverable).Append(',')
                  .Append(family.TriangleInteriorShadow).Append(',')
                  .Append(family.TrianglePureBoundaryCandidate).Append(',')
                  .Append(family.TriangleBoundaryRecoverable).Append(',')
                  .Append(family.TriangleBoundaryShadow).Append(',')
                  .Append(family.TriangleBoundaryBlockedExactMissing).Append(',')
                  .Append(family.TriangleBoundaryShadowUnexpected).Append(',')
                  .Append(family.TriangleBoundaryCommitted).Append(',')
                  .Append(family.BoundaryCommittedPages).Append(',')
                  .Append(family.TriangleBoundaryBlockedCoverage).Append(',')
                  .Append(family.TriangleBoundaryBlockedHard).Append(',')
                  .Append(family.TriangleBoundaryBlockedMulti).Append(',')
                  .Append(family.TriangleBoundaryBlockedTransition).Append(',')
                  .Append(family.DecisionReason).Append(',')
                  .Append(family.Finalized ? 1 : 0).Append(',')
                  .Append(family.Swapped ? 1 : 0).Append(',')
                  .Append(family.MixedCommitted ? 1 : 0).Append(',')
                  .Append(family.MixedCommittedPages).Append(',')
                  .Append(family.MixedCommittedTriangles).AppendLine();
            }
            sb.AppendLine();
            sb.AppendLine("child16_triangle_counterfactual_csv:");
            sb.AppendLine("parent_x,parent_y,parent_z,child_x,child_y,child_z,suspected_center_false,interior_shadow_triangles,pure_boundary,interior_recoverable,boundary_exact_recoverable,boundary_shadow_triangles,boundary_exact_missing,boundary_shadow_unexpected,boundary_committed_triangles,legacy_would_block_coverage,legacy_would_block_hard,legacy_would_block_multi,legacy_would_block_transition,real_edge,ambiguous_normal,ambiguous_weight,ambiguous_multi,transition");
            foreach (Family family in _families.Values)
            {
                for (int i = 0; i < family.TrianglePages.Count; i++)
                {
                    TriangleCounterfactualPage page = family.TrianglePages[i];
                    sb.Append(family.Parent.x).Append(',')
                      .Append(family.Parent.y).Append(',')
                      .Append(family.Parent.z).Append(',')
                      .Append(page.Coordinate.x).Append(',')
                      .Append(page.Coordinate.y).Append(',')
                      .Append(page.Coordinate.z).Append(',')
                      .Append(page.SuspectedCenterFalse).Append(',')
                      .Append(page.InteriorShadowTriangles).Append(',')
                      .Append(page.PureBoundary).Append(',')
                      .Append(page.InteriorRecoverable).Append(',')
                      .Append(page.BoundaryRecoverable).Append(',')
                      .Append(page.BoundaryShadowTriangles).Append(',')
                      .Append(page.BoundaryBlockedExactMissing).Append(',')
                      .Append(page.BoundaryShadowUnexpected).Append(',')
                      .Append(page.BoundaryCommittedTriangles).Append(',')
                      .Append(page.BoundaryBlockedCoverage).Append(',')
                      .Append(page.BoundaryBlockedHard).Append(',')
                      .Append(page.BoundaryBlockedMulti).Append(',')
                      .Append(page.BoundaryBlockedTransition).Append(',')
                      .Append(page.RealEdge).Append(',')
                      .Append(page.AmbiguousNormal).Append(',')
                      .Append(page.AmbiguousWeight).Append(',')
                      .Append(page.AmbiguousMulti).Append(',')
                      .Append(page.Transition).AppendLine();
                }
            }
            sb.AppendLine();
            sb.AppendLine("[parent32]");
            _parent32.AppendStaticReplayReport(sb);
            sb.AppendLine();
            sb.AppendLine("[child16]");
            _child16.AppendStaticReplayReport(sb);
        }

        private static int Flatten(int3 coordinate, int3 grid)
        {
            return coordinate.x + grid.x * (coordinate.y + grid.y * coordinate.z);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _parent32.StaticReplayPageCommitted -= OnParentCommitted;
            _child16.StaticReplayPageCommitted -= OnChildCommitted;
            _parent32.Dispose();
            _child16.Dispose();
            _families.Clear();
            _childToParent.Clear();
            _ledger64.Clear();
        }
    }
}
