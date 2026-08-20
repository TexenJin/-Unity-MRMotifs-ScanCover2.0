using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Hands;

namespace Genesis.RoomScan
{
    /// <summary>
    /// 手部语义剔除球采集器（08-20，纯数据类，DepthCapture 持有，每预处理拍 Refresh 一次）。
    ///
    /// 与平台原生手部剔除（XR_META_environment_depth_hand_removal）分工：
    ///  - 裸手、手追在跑 → 原生剔除生效（真手形分割+背景估计填充，优于球近似），
    ///    本采集器 skipHandJoints=true，只补控制器球，避免把原生填的背景再打回弃权。
    ///  - 持握控制器 → Quest 手追暂停（multimodal 未开）→ 原生剔除自动失效 →
    ///    控制器球是唯一覆盖（用户拍板同车）：柄+握持手本是一个物理团块，一球罩住。
    ///  - 原生剔除不可用（老系统/不支持）→ skipHandJoints=false，手关节球退化为兜底。
    ///
    /// 位姿统一为 session tracking space → 由 DepthCapture 传入的 TrackingSpace 变换转世界，
    /// 与深度管线（TryGetPoses 同为 session tracking space）坐标系严格一致。
    /// 编辑器/无 XR：两路皆空 → Count=0 → 打码 pass 整体跳过，零成本零风险。
    /// </summary>
    public sealed class HandSphereCollector
    {
        public const int MaxSpheres = 48;

        private readonly Vector4[] _spheres = new Vector4[MaxSpheres];
        /// <summary>球数组（xyz=世界球心，w=半径米）；仅前 Count 个有效。</summary>
        public Vector4[] Spheres => _spheres;
        /// <summary>本拍有效球数。</summary>
        public int Count { get; private set; }
        /// <summary>手部追踪子系统是否在跑（HUD 诊断：手0 时区分"没手"还是"手追没跑"）。</summary>
        public bool HandTrackingRunning { get; private set; }

        private static readonly List<XRHandSubsystem> s_subsystems = new List<XRHandSubsystem>(2);

        // 每手 17 球全指链（08-20 护甲补全，录屏 025342 判决：5 球版=筛子——无名指/小指
        // 全裸、指中节裸奔，手撑床/墙时贴面的恰是裸指节→撑起幻影块）。
        // 三档半径：0=掌/腕大球，1=指节中球（近节/中节/拇指远节），2=指尖小球。
        // 半径偏保守（比真实手指粗数倍）：吸收手追抖动+球近似形状误差；
        // 代价=贴墙指缝正后方一小片墙像素同被弃权，手挪开下一拍即恢复，可忽略。
        private const int RADIUS_PALM = 0, RADIUS_KNUCKLE = 1, RADIUS_TIP = 2;
        private static readonly (XRHandJointID joint, int radius)[] s_joints =
        {
            (XRHandJointID.Palm, RADIUS_PALM),
            (XRHandJointID.Wrist, RADIUS_PALM),
            (XRHandJointID.IndexProximal, RADIUS_KNUCKLE),
            (XRHandJointID.IndexIntermediate, RADIUS_KNUCKLE),
            (XRHandJointID.IndexTip, RADIUS_TIP),
            (XRHandJointID.MiddleProximal, RADIUS_KNUCKLE),
            (XRHandJointID.MiddleIntermediate, RADIUS_KNUCKLE),
            (XRHandJointID.MiddleTip, RADIUS_TIP),
            (XRHandJointID.RingProximal, RADIUS_KNUCKLE),
            (XRHandJointID.RingIntermediate, RADIUS_KNUCKLE),
            (XRHandJointID.RingTip, RADIUS_TIP),
            (XRHandJointID.LittleProximal, RADIUS_KNUCKLE),
            (XRHandJointID.LittleIntermediate, RADIUS_KNUCKLE),
            (XRHandJointID.LittleTip, RADIUS_TIP),
            (XRHandJointID.ThumbProximal, RADIUS_KNUCKLE),
            (XRHandJointID.ThumbDistal, RADIUS_KNUCKLE),
            (XRHandJointID.ThumbTip, RADIUS_TIP),
        };

        /// <summary>
        /// 刷新球列表。trackingSpace=null 时位姿无法转世界，直接清空（打码跳过）。
        /// skipHandJoints=true（原生剔除生效中）时只采控制器球。
        /// </summary>
        public void Refresh(Transform trackingSpace, bool skipHandJoints,
            float palmRadius, float tipRadius, float controllerRadius)
        {
            Count = 0;
            HandTrackingRunning = false;
            if (trackingSpace == null) return;

            if (!skipHandJoints)
            {
                SubsystemManager.GetSubsystems(s_subsystems);
                for (int s = 0; s < s_subsystems.Count; s++)
                {
                    var sub = s_subsystems[s];
                    if (sub == null || !sub.running) continue;
                    HandTrackingRunning = true;
                    CollectHand(sub.leftHand, trackingSpace, palmRadius, tipRadius);
                    CollectHand(sub.rightHand, trackingSpace, palmRadius, tipRadius);
                    if (Count >= MaxSpheres) return;
                }
            }

            // 控制器球：追踪中即补。OVRManager 不在时 OVRInput 静默返回 false，安全。
            CollectController(OVRInput.Controller.LTouch, trackingSpace, controllerRadius);
            CollectController(OVRInput.Controller.RTouch, trackingSpace, controllerRadius);
        }

        private void CollectHand(XRHand hand, Transform trackingSpace, float palmRadius, float tipRadius)
        {
            float knuckleRadius = (palmRadius + tipRadius) * 0.5f; // 指节中球=掌/指尖的几何中值
            for (int j = 0; j < s_joints.Length && Count < MaxSpheres - 2; j++)
            {
                var joint = hand.GetJoint(s_joints[j].joint);
                if (!joint.TryGetPose(out Pose pose)) continue; // 未追踪/无数据：该关节缺席，其余指链照常
                float r = s_joints[j].radius == RADIUS_PALM ? palmRadius
                        : s_joints[j].radius == RADIUS_TIP ? tipRadius : knuckleRadius;
                Vector3 world = trackingSpace.TransformPoint(pose.position);
                _spheres[Count++] = new Vector4(world.x, world.y, world.z, r);
            }
        }

        private void CollectController(OVRInput.Controller ctrl, Transform trackingSpace, float radius)
        {
            if (Count >= MaxSpheres) return;
            if (!OVRInput.GetControllerPositionTracked(ctrl)) return;
            Vector3 world = trackingSpace.TransformPoint(OVRInput.GetLocalControllerPosition(ctrl));
            _spheres[Count++] = new Vector4(world.x, world.y, world.z, radius);
        }
    }
}
