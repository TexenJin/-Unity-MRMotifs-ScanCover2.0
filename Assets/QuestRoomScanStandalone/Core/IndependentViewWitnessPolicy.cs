using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// 生产探针与影子赛马共用的“独立观察”几何语义。容量是调用方账本的
    /// 属性，不属于几何独立性本身；这样离线可以区分“视角相关”与
    /// “证词有效但账本已满”。本类不拥有任何生产决策权限。
    /// </summary>
    internal static class IndependentViewWitnessPolicy
    {
        internal const float BaselineMetres = 0.08f;
        internal const float AngleDeg = 3f;
        internal const int FrameGap = 2;

        internal static void ComparePair(Vector3 previousCamera,
            Vector3 previousDirection, int previousFrame, Vector3 camera,
            Vector3 direction, int frame, out float baselineMetres,
            out float spreadDeg, out bool timeSeparated,
            out bool poseSeparated)
        {
            baselineMetres = Vector3.Distance(previousCamera, camera);
            spreadDeg = Vector3.Angle(previousDirection, direction);
            timeSeparated = frame - previousFrame >= FrameGap;
            poseSeparated = baselineMetres >= BaselineMetres ||
                            spreadDeg >= AngleDeg;
        }

        internal static IndependentWitnessDecision ClassifyBlocked(
            bool blockedByTime, bool blockedByPose)
        {
            if (blockedByTime && blockedByPose)
                return IndependentWitnessDecision.CorrelatedTimeAndPose;
            if (blockedByTime)
                return IndependentWitnessDecision.CorrelatedTime;
            if (blockedByPose)
                return IndependentWitnessDecision.CorrelatedPose;
            return IndependentWitnessDecision.Accepted;
        }

        internal static string Name(IndependentWitnessDecision decision)
        {
            switch (decision)
            {
                case IndependentWitnessDecision.Accepted:
                    return "accepted";
                case IndependentWitnessDecision.CorrelatedTime:
                    return "correlated_time";
                case IndependentWitnessDecision.CorrelatedPose:
                    return "correlated_pose";
                case IndependentWitnessDecision.CorrelatedTimeAndPose:
                    return "correlated_time_and_pose";
                case IndependentWitnessDecision.CapacityFull:
                    return "capacity_full";
                default:
                    return "invalid_capacity";
            }
        }
    }

    internal enum IndependentWitnessDecision
    {
        Accepted = 0,
        CorrelatedTime = 1,
        CorrelatedPose = 2,
        CorrelatedTimeAndPose = 3,
        CapacityFull = 4,
        InvalidCapacity = 5
    }
}
