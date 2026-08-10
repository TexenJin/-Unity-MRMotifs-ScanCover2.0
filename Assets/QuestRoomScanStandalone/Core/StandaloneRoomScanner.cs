using System;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// QRS 房间扫描链的独立测试编排器（裁剪版 RoomScanner）。
    /// 只保留：深度采集 → TSDF 融合 → GPU Surface Nets 提取 → 顶点色网格渲染。
    /// 砍掉：持久化、锚点、纹理精炼、三平面、关键帧、GSplat、房间理解、调试菜单。
    ///
    /// 所有兄弟组件挂在同一个 GameObject 上自动解析。
    /// 输入由 <see cref="StandaloneScanInput"/> 处理：
    ///   右手柄扳机 = 开始/继续扫描，A = 暂停，B = 保存累计账并清空重扫。
    /// </summary>
    [RequireComponent(typeof(DepthCapture), typeof(VolumeIntegrator), typeof(MeshExtractor))]
    public class StandaloneRoomScanner : MonoBehaviour
    {
        public static StandaloneRoomScanner Instance { get; private set; }

        [Header("扫描频率")]
        [SerializeField] private float integrationHz = 30f;
        [SerializeField] private float meshExtractionHz = 30f;

        [Header("渲染")]
        [SerializeField, Tooltip("主显示形态：开=线框（QRS Wireframe，重心坐标边缘检测）；关=顶点色实体（QRS Vertex）")]
        private bool wireframeMode = true;
        [SerializeField, Range(0.2f, 5f), Tooltip("线框模式的线条粗细倍率（1.0 对齐原 SC 工程细线观感，可按需调）")]
        private float wireThickness = 1.0f;

        [Header("覆盖范围")]
        [SerializeField, Tooltip("头部排除区（QRS 原版防自扫）：开=头周圆柱内永不生成网格（半径在 VolumeIntegrator.exclusionRadius 调）；关=周围近距也能覆盖网格")]
        private bool enableHeadExclusion = true;

        /// <summary>当前是否线框显示。</summary>
        public bool IsWireframe => wireframeMode;

        /// <summary>在线框 / 顶点色实体之间切换（QRS SetRenderMode 的二态精简版）。</summary>
        public void ToggleWireframe()
        {
            wireframeMode = !wireframeMode;
            ApplyDisplayMode();
            NotifyInput(wireframeMode ? "切到线框" : "切到实体");
        }

        [Header("日志")]
        [SerializeField] private LogLevel logLevel = LogLevel.Info;

        private DepthCapture _depthCapture;
        private VolumeIntegrator _volumeIntegrator;
        private MeshExtractor _meshExtractor;
        private PassthroughCameraProvider _cameraProvider;

        /// <summary>正在融合（未暂停）。</summary>
        public bool IsScanning { get; private set; }
        /// <summary>本次运行期间曾经开始过扫描（暂停后为 true，清空后归 false）。</summary>
        public bool HasStarted { get; private set; }

        public event Action ScanStarted;
        public event Action ScanStopped;
        /// <summary>每次深度融合后触发。</summary>
        public event Action Integrated;
        /// <summary>每次网格提取后触发。</summary>
        public event Action MeshExtracted;

        public VolumeIntegrator VolumeIntegrator => _volumeIntegrator;
        public DepthCapture DepthCapture => _depthCapture;
        public MeshExtractor MeshExtractor => _meshExtractor;

        private float _lastIntegrationTime;
        private float _lastMeshTime;
        private float _lastScannerLog;
        private int _integrateCount;

        private float IntegrationInterval => 1f / integrationHz;
        private float MeshInterval => 1f / meshExtractionHz;

        // ─────────────────────────────────────────────────────────────
        //  生命周期
        // ─────────────────────────────────────────────────────────────

        private void Awake()
        {
            Instance = this;
            // Runtime lock: serialized scene values cannot accidentally revive
            // the in-headset HUD or the diagnostic ROI frame.
            showDebugHud = false;
            showDiagnosticRoiFrame = false;
            Logger.Level = logLevel;
            _depthCapture = GetComponent<DepthCapture>();
            _volumeIntegrator = GetComponent<VolumeIntegrator>();
            _meshExtractor = GetComponent<MeshExtractor>();
            _cameraProvider = GetComponent<PassthroughCameraProvider>();
            SetSafeShaderDefaults();
        }

        private void Start()
        {
            if (!XRRuntimeGuard.IsXRActive)
            {
                Logger.Warning("StandaloneRoomScanner: " + XRRuntimeGuard.EditorDisabledMessage);
                enabled = false;
                return;
            }

            if (enableHeadExclusion) SetupHeadExclusion();
            if (showDepthPointCloud) gameObject.AddComponent<DepthPointCloudOverlay>();
            StartCoroutine(ConfigureCameraForPassthrough());
            if (showDebugHud)
                StartCoroutine(CreateHudWhenCameraReady());
            Application.logMessageReceived += OnLogMessage;
            Logger.Info("QRS 独立链就绪 — 右手柄扳机开始扫描，A 暂停，B 停止清空");
        }

        /// <summary>
        /// OVRCameraRig 运行时自动创建的中眼相机默认是 Skybox 清除 + 不透明背景，
        /// 会挡住底层透视画面。等 Camera.main 出现后改成纯色透明黑清除。
        /// </summary>
        private System.Collections.IEnumerator ConfigureCameraForPassthrough()
        {
            while (Camera.main == null)
                yield return null;

            var cam = Camera.main;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0f, 0f, 0f, 0f);
            cam.nearClipPlane = 0.1f;
            Logger.Info($"相机已配置为透视底层模式: {cam.gameObject.name}");
        }

        // ─────────────────────────────────────────────────────────────
        //  头显内中文状态面板（世界空间 UI，不依赖 adb）
        // ─────────────────────────────────────────────────────────────

        [Header("调试面板")]
        [SerializeField] private bool showDebugHud = false;
        [SerializeField, Tooltip("在面板右上角开一个当前深度实时预览小窗（青=近 绿=中 红=远 暗=无效）。" +
            "用途：盯着幽灵网格时看深度画面里那个斑块还在不在——在=深度自洽幻觉（Meta侧时序锁定）；转头后斑块从预览消失=深度刷新")]
        private bool showDepthPreview = true;
        [SerializeField, Tooltip("世界空间实时深度点云叠加层：当前深度以 3D 点云叠在网格上同屏对照。" +
            "幽灵位置有点云覆盖=深度自洽幻觉；空空如也却有网格=矛盾在我们侧")]
        private bool showDepthPointCloud = true;
        [SerializeField, Tooltip("显示只读断崖样本框：左侧近面、中间边缘、右侧远景。只影响诊断计数与导出。")]
        private bool showDiagnosticRoiFrame = false;

        private UnityEngine.UI.Text _hudText;
        private RectTransform _hudRect;
        private GameObject _diagnosticRoiFrame;
        private string _hudStatus = "就绪";
        private string _hudLastError = "";
        private string _hudLastInput = "无";
        private float _hudRefreshTimer;

        /// <summary>输入处理器回显：最近一次识别到的按键（用于区分"输入没到达"与"启动失败"）。</summary>
        public void NotifyInput(string what)
        {
            _hudLastInput = $"{what} ({Time.time:F0}s)";
        }

        private System.Collections.IEnumerator CreateHudWhenCameraReady()
        {
            while (Camera.main == null)
                yield return null;

            Font font = null;
            try { font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf"); } catch { }
            if (font == null)
            {
                try { font = Resources.GetBuiltinResource<Font>("Arial.ttf"); } catch { }
            }

            var canvasGo = new GameObject("[QRS] 状态面板");
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            _hudRect = canvasGo.GetComponent<RectTransform>();
            // Keep approximately the same physical width as the old panel, but add
            // vertical room for the diagnostic ledgers so wrapped rows remain visible.
            _hudRect.sizeDelta = new Vector2(1080f, 650f);
            canvasGo.transform.localScale = Vector3.one * 0.00072f;

            // 置顶材质：WorldSpace Canvas 走 UI/Default 时 ZTest=LEqual 会被网格/墙面挡住，
            // 换 QRS/HUDAlwaysOnTop（ZTest Always + Overlay 队列）。Resources 加载防裁剪，找不到静默回退。
            Material hudMat = null;
            var hudShader = Resources.Load<Shader>("HUDAlwaysOnTop");
            if (hudShader != null) hudMat = new Material(hudShader);

            var textGo = new GameObject("Text");
            textGo.transform.SetParent(canvasGo.transform, false);
            _hudText = textGo.AddComponent<UnityEngine.UI.Text>();
            _hudText.font = font;
            _hudText.fontSize = 28;
            _hudText.lineSpacing = 0.88f;
            _hudText.color = Color.white;
            _hudText.alignment = TextAnchor.UpperLeft;
            if (hudMat != null) _hudText.material = hudMat;
            // Wrap：开预览时文字区收窄到左侧，长行折行而不是溢到小窗底下
            _hudText.horizontalOverflow = HorizontalWrapMode.Wrap;
            _hudText.verticalOverflow = VerticalWrapMode.Overflow;
            var rt = textGo.GetComponent<RectTransform>();
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = new Vector2(18f, 12f);
            // 开深度预览时右侧留出 320px 给小窗，文字不压图
            rt.offsetMax = new Vector2(showDepthPreview ? -258f : -18f, -12f);

            // 半透明黑底，保证在透视画面上可读
            var bgGo = new GameObject("Bg");
            bgGo.transform.SetParent(canvasGo.transform, false);
            bgGo.transform.SetAsFirstSibling();
            var img = bgGo.AddComponent<UnityEngine.UI.Image>();
            img.color = new Color(0f, 0f, 0f, 0.55f);
            if (hudMat != null) img.material = hudMat;
            var brt = bgGo.GetComponent<RectTransform>();
            brt.anchorMin = Vector2.zero;
            brt.anchorMax = Vector2.one;
            brt.offsetMin = Vector2.zero;
            brt.offsetMax = Vector2.zero;

            if (showDepthPreview) CreateDepthPreview(canvasGo.transform);
            if (showDiagnosticRoiFrame && _meshExtractor != null && _meshExtractor.DiagnosticRoiEnabled)
                CreateDiagnosticRoiFrame(Camera.main, font, hudMat);

            Logger.Info("调试面板已创建");
        }

        /// <summary>
        /// 深度实时预览小窗：RawImage 挂 QRS/DepthPreview 材质，
        /// shader 直接采样全局 gsDepthTex，零 C# 侧每帧拷贝，零额外接线。
        /// shader 在 Resources 下（防打包裁剪），找不到就静默跳过不影响面板。
        /// </summary>
        /// <summary>
        /// 只读断崖样本框。框与计算着色器使用同一组归一化范围：
        /// 左栏放近面，中间窄栏压住断崖边，右栏放远景。
        /// 该 Canvas 不参与射线、融合、提取或准入，只给操作者对准诊断 ROI。
        /// </summary>
        private void CreateDiagnosticRoiFrame(Camera cam, Font font, Material hudMat)
        {
            if (cam == null || _meshExtractor == null) return;

            var go = new GameObject("[QRS] 断崖诊断框");
            _diagnosticRoiFrame = go;
            go.transform.SetParent(cam.transform, false);

            var canvas = go.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.overrideSorting = true;
            canvas.sortingOrder = 32760;

            const float distance = 1f;
            const float canvasWidthPx = 1000f;
            float viewHeight = 2f * Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad) * distance;
            float viewWidth = viewHeight * Mathf.Max(0.1f, cam.aspect);
            float canvasHeightPx = canvasWidthPx * viewHeight / Mathf.Max(0.001f, viewWidth);

            var root = go.GetComponent<RectTransform>();
            root.sizeDelta = new Vector2(canvasWidthPx, canvasHeightPx);
            go.transform.localPosition = Vector3.forward * distance;
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale = Vector3.one * (viewWidth / canvasWidthPx);

            Vector4 r = _meshExtractor.DiagnosticRoiRect;
            float x0 = Mathf.Clamp01(Mathf.Min(r.x, r.z));
            float y0 = Mathf.Clamp01(Mathf.Min(r.y, r.w));
            float x1 = Mathf.Clamp01(Mathf.Max(r.x, r.z));
            float y1 = Mathf.Clamp01(Mathf.Max(r.y, r.w));
            Vector2 splits = _meshExtractor.DiagnosticRoiSplitX;
            float sx0 = Mathf.Clamp(splits.x, x0, x1);
            float sx1 = Mathf.Clamp(splits.y, sx0, x1);

            Color outer = new Color(1f, 1f, 1f, 0.85f);
            Color divider = new Color(1f, 0.82f, 0.18f, 0.9f);
            const float linePx = 3f;

            AddDiagnosticRoiLine(go.transform, "左边", new Vector2(x0, y0), new Vector2(x0, y1),
                new Vector2(linePx, 0f), outer, hudMat);
            AddDiagnosticRoiLine(go.transform, "右边", new Vector2(x1, y0), new Vector2(x1, y1),
                new Vector2(linePx, 0f), outer, hudMat);
            AddDiagnosticRoiLine(go.transform, "下边", new Vector2(x0, y0), new Vector2(x1, y0),
                new Vector2(0f, linePx), outer, hudMat);
            AddDiagnosticRoiLine(go.transform, "上边", new Vector2(x0, y1), new Vector2(x1, y1),
                new Vector2(0f, linePx), outer, hudMat);
            AddDiagnosticRoiLine(go.transform, "近面边界", new Vector2(sx0, y0), new Vector2(sx0, y1),
                new Vector2(linePx, 0f), divider, hudMat);
            AddDiagnosticRoiLine(go.transform, "远景边界", new Vector2(sx1, y0), new Vector2(sx1, y1),
                new Vector2(linePx, 0f), divider, hudMat);

            AddDiagnosticRoiLabel(go.transform, "近面", (x0 + sx0) * 0.5f, y1, font, hudMat);
            AddDiagnosticRoiLabel(go.transform, "边缘", (sx0 + sx1) * 0.5f, y1, font, hudMat);
            AddDiagnosticRoiLabel(go.transform, "远景", (sx1 + x1) * 0.5f, y1, font, hudMat);
        }

        private static void AddDiagnosticRoiLine(
            Transform parent, string name, Vector2 anchorMin, Vector2 anchorMax,
            Vector2 sizeDelta, Color color, Material material)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var image = go.AddComponent<UnityEngine.UI.Image>();
            image.color = color;
            image.raycastTarget = false;
            if (material != null) image.material = material;
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            rt.anchoredPosition = Vector2.zero;
            rt.sizeDelta = sizeDelta;
        }

        private static void AddDiagnosticRoiLabel(
            Transform parent, string label, float x, float y, Font font, Material material)
        {
            var go = new GameObject(label);
            go.transform.SetParent(parent, false);
            var text = go.AddComponent<UnityEngine.UI.Text>();
            text.text = label;
            text.font = font;
            text.fontSize = 28;
            text.fontStyle = FontStyle.Bold;
            text.alignment = TextAnchor.UpperCenter;
            text.color = new Color(1f, 0.9f, 0.35f, 0.95f);
            text.raycastTarget = false;
            if (material != null) text.material = material;
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(x, y);
            rt.anchorMax = new Vector2(x, y);
            rt.pivot = new Vector2(0.5f, 1f);
            rt.anchoredPosition = new Vector2(0f, -8f);
            rt.sizeDelta = new Vector2(130f, 42f);
        }

        private void CreateDepthPreview(Transform parent)
        {
            var shader = Resources.Load<Shader>("DepthPreview");
            if (shader == null)
            {
                Logger.Warning("深度预览 shader 未找到（Resources/DepthPreview），小窗跳过");
                return;
            }

            var go = new GameObject("DepthPreview");
            go.transform.SetParent(parent, false);
            var raw = go.AddComponent<UnityEngine.UI.RawImage>();
            raw.texture = Texture2D.whiteTexture; // 防空纹理剔除，shader 实际采样全局 gsDepthTex
            raw.material = new Material(shader);
            raw.raycastTarget = false;
            var prt = go.GetComponent<RectTransform>();
            // 右上角贴边，文字区已收窄折行，互不遮挡
            prt.anchorMin = new Vector2(1f, 1f);
            prt.anchorMax = new Vector2(1f, 1f);
            prt.pivot = new Vector2(1f, 1f);
            prt.anchoredPosition = new Vector2(-12f, -12f);
            prt.sizeDelta = new Vector2(228f, 128f);
        }

        private void UpdateHud()
        {
            if (_hudText == null) return;

            var cam = Camera.main;
            if (cam != null)
            {
                var ct = cam.transform;
                Vector3 targetPos = ct.position + ct.forward * 1.1f + Vector3.down * 0.18f;
                _hudRect.position = Vector3.Lerp(_hudRect.position, targetPos, Time.deltaTime * 4f);
                _hudRect.rotation = Quaternion.LookRotation(
                    _hudRect.position - ct.position, Vector3.up);
            }

            _hudRefreshTimer += Time.deltaTime;
            if (_hudRefreshTimer < 0.25f) return;
            _hudRefreshTimer = 0f;

            int verts = _meshExtractor != null ? _meshExtractor.LastVertexCount : 0;
            int tris = _meshExtractor != null ? _meshExtractor.LastIndexCount / 3 : 0;
            int integrated = _volumeIntegrator != null ? _volumeIntegrator.IntegrationCount : 0;
            bool camPlaying = _cameraProvider != null && _cameraProvider.IsPlaying;

            string edgeStat = "—";
            if (_depthCapture != null && _depthCapture.HasEdgeCleanStats)
            {
                uint ec = _depthCapture.LastEdgeCleanCount;
                uint gz = _depthCapture.LastGrazingKillCount;
                edgeStat = (ec >= 10000u ? (ec / 10000f).ToString("0.0") + "万" : ec.ToString()) +
                           " 掠:" + (gz >= 10000u ? (gz / 10000f).ToString("0.0") + "万" : gz.ToString());
            }

            _hudText.text =
                $"【QRS独立链】{_hudStatus}\n" +
                $"深度:{(DepthCapture.DepthAvailable ? "可用" : "无")}#{(_depthCapture != null ? _depthCapture.FrameCount : 0)}  相机:{(camPlaying ? "运行" : "未运行")}  融合:{integrated}帧\n" +
                $"顶点:{verts}  三角:{tris}\n" +
                $"矛盾票:{(_volumeIntegrator != null ? _volumeIntegrator.GetCarveStatsCompact() : "无")}" +
                $" 角速:{(_volumeIntegrator != null ? _volumeIntegrator.SmoothedAngularSpeed : 0f):F0}°/s" +
                $" 缘:{edgeStat}\n" +
                $"供料账:{(_volumeIntegrator != null ? _volumeIntegrator.GetSupplyLedgerCompact() : "无")}\n" +
                $"断层影:{(_volumeIntegrator != null ? _volumeIntegrator.GetAdaptiveGapShadowCompact() : "无")}\n" +
                $"边缘源:{(_volumeIntegrator != null ? _volumeIntegrator.GetEdgeSourceLedgerCompact() : "无")}\n" +
                $"供体证:{(_volumeIntegrator != null ? _volumeIntegrator.GetDilationDonorLedgerCompact() : "无")}\n" +
                $"路径证:{(_volumeIntegrator != null ? _volumeIntegrator.GetDilationPathLedgerCompact() : "无")}\n" +
                $"接力账:{(_volumeIntegrator != null ? _volumeIntegrator.GetDilationRelayLedgerCompact() : "无")}\n" +
                $"准入来源:{(_meshExtractor != null ? _meshExtractor.GetAdmissionSourceStatsCompact() : "无")}\n" +
                $"真实确认:{(_meshExtractor != null ? _meshExtractor.GetRealConfirmationStatsCompact() : "无")}\n" +
                $"双轨视图:{(_meshExtractor != null ? _meshExtractor.GetJointDiagnosticStatsCompact() : "无")}\n" +
                $"累计账:{(_meshExtractor != null ? _meshExtractor.GetLedgerSessionStatsCompact() : "无")}\n" +
                $"框内三栏:{(_meshExtractor != null ? _meshExtractor.GetDiagnosticRoiStatsCompact() : "无")}\n" +
                $"提取:生产  严格:{(_meshExtractor != null ? _meshExtractor.GetStrictObservedStatsCompact() : "无")}\n" +
                $"最近按键:{_hudLastInput}\n" +
                BuildInputDiagLine() +
                $"显示:{(wireframeMode ? "线框" : "实体")}  扳机=开始/继续  A=暂停  B=保存并清空\n" +
                $"摇杆按=线框  Y=生产A/候选B" +
                (_hudLastError.Length > 0 ? $"\n<color=#FF6060>错误:{_hudLastError}</color>" : "");
        }

        /// <summary>
        /// 输入链路体检行：OVRManager 是否存活、系统认为接着什么控制器、扳机模拟量实时值。
        /// OVRInput 只有在 OVRManager 存活时才会每帧更新，GetDown/Get 才有意义。
        /// </summary>
        private static string BuildInputDiagLine()
        {
            bool mgrAlive = OVRManager.instance != null;
            var connected = OVRInput.GetConnectedControllers();
            float trig = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.RTouch);
            bool touchTracked = OVRInput.GetControllerPositionTracked(OVRInput.Controller.RTouch);
            bool btnA = OVRInput.Get(OVRInput.Button.One, OVRInput.Controller.RTouch);
            bool btnB = OVRInput.Get(OVRInput.Button.Two, OVRInput.Controller.RTouch);
            return $"输入:管理器{(mgrAlive ? "存活" : "缺失")} 连接:{connected} 右手追踪:{(touchTracked ? "有" : "无")} 扳机量:{trig:F2} A:{(btnA ? "1" : "0")} B:{(btnB ? "1" : "0")}\n";
        }

        private void OnLogMessage(string condition, string stackTrace, LogType type)
        {
            if (type == LogType.Exception || type == LogType.Error)
                _hudLastError = condition.Length > 80 ? condition.Substring(0, 80) : condition;
        }

        private void OnDestroy()
        {
            Application.logMessageReceived -= OnLogMessage;
            if (_diagnosticRoiFrame != null)
                Destroy(_diagnosticRoiFrame);
        }

        private void OnDisable()
        {
            if (IsScanning) PauseScanning();
        }

        private void Update()
        {
            UpdateHud();

            if (!IsScanning || !DepthCapture.DepthAvailable) return;

            float t = Time.time;

            if (t - _lastIntegrationTime >= IntegrationInterval)
            {
                _lastIntegrationTime = t;

                ProvideColorFrame();
                _volumeIntegrator.Integrate();
                Integrated?.Invoke();
                _integrateCount++;

                if (t - _lastMeshTime >= MeshInterval)
                {
                    _lastMeshTime = t;
                    _meshExtractor.Extract();
                    MeshExtracted?.Invoke();
                }
            }

            if (t - _lastScannerLog >= 5f)
            {
                _lastScannerLog = t;
                Logger.Verbose($"扫描中: 融合次数={_integrateCount}, 深度可用={DepthCapture.DepthAvailable}");
            }
        }

        // ─────────────────────────────────────────────────────────────
        //  公开 API（输入处理器调用）
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// 开始（或暂停后继续）深度融合与网格提取。
        /// 保留 QRS 原版的异步分段启动：先提交 GPU 大资源（TSDF ~150MB +
        /// Surface Nets ~480MB），跨帧 yield 两次后再启用透视相机与深度，
        /// 避免 PCA 硬件缓冲队列握手与计算调度同帧竞争导致 Vulkan 卡死。
        /// 继续扫描时 ReallocateVolumes / EnsureInitialized 均为幂等 no-op，
        /// 已有体积数据保持不变。
        /// </summary>
        public async Task StartScanningAsync()
        {
            if (IsScanning) return;
            IsScanning = true;
            try
            {
                // 阶段 1：GPU 体积 bring-up
                _volumeIntegrator.ReallocateVolumes();
                await Task.Yield();
                await Task.Yield();

                // 阶段 2：Surface Nets 提取器
                _meshExtractor.EnsureInitialized();
                await Task.Yield();
                await Task.Yield();

                float t = Time.time;
                _lastIntegrationTime = t;
                _lastMeshTime = t;
                _cameraAvailable = false;

                // 阶段 3：相机 + 深度（此时启动安全）
                _cameraProvider?.StartCapture();
                _depthCapture.StartDepthCapture();

                bool resuming = HasStarted;
                if (!resuming)
                    _meshExtractor.BeginLedgerSession();
                HasStarted = true;
                _hudStatus = resuming ? "扫描中(继续)" : "扫描中";
                _hudLastError = "";
                Logger.Info($"开始扫描 — 继续上次={resuming}, 已累计融合={_volumeIntegrator.IntegrationCount}");
                ScanStarted?.Invoke();
            }
            catch (Exception e)
            {
                // 重置重入保护，允许用户重试
                IsScanning = false;
                _hudStatus = "启动失败";
                _hudLastError = e.Message;
                throw;
            }
        }

        /// <summary>暂停融合与相机，体积与网格数据保留，可再次开始继续。</summary>
        public void PauseScanning()
        {
            if (!IsScanning) return;
            IsScanning = false;

            _cameraProvider?.StopCapture();
            _depthCapture.StopDepthCapture();

            Logger.Info("已暂停 — 扳机继续，B 保存累计账并清空");
            _hudStatus = "已暂停";
            ScanStopped?.Invoke();
        }

        /// <summary>
        /// 将本次累计账导出后，原子式清空 TSDF、网格、融合计数与记账会话。
        /// A 暂停不会触发这里；下一次扳机将开始全新扫描和全新账簿。
        /// </summary>
        public void StopAndClearScan()
        {
            bool wasActive = IsScanning || HasStarted;
            PauseScanning();
            string ledgerPath = _meshExtractor != null
                ? _meshExtractor.FinalizeAndExportLedger("B键保存并清空")
                : "";

            // B is a save-then-clear transaction.  If persistence failed, keep
            // the volume, mesh and open ledger intact so the user can retry.
            if (wasActive && string.IsNullOrEmpty(ledgerPath))
            {
                _hudStatus = "账簿保存失败，未清空";
                Logger.Error("累计账簿保存失败：已中止清空，扫描数据仍保留");
                return;
            }

            // Invalidate outstanding GPU readbacks before touching the volume;
            // otherwise an old callback can repopulate a freshly cleared HUD.
            _meshExtractor?.ResetLedgerSessionAfterClear();
            HasStarted = false;
            _integrateCount = 0;

            _volumeIntegrator.Clear();
            _volumeIntegrator.ResetSessionCounters();
            if (_meshExtractor.IsInitialized)
                _meshExtractor.Reinitialize(); // 同时重置时序混合状态

            if (wasActive)
                Logger.Info($"已保存并清空 — 账簿={ledgerPath}；扳机开始全新扫描");
            _hudStatus = ledgerPath.Length > 0 ? "已保存并清空" : "已清空(账簿为空)";
        }

        /// <summary>严格生产网格已锁定；保留入口仅为旧输入/场景兼容。</summary>
        public void ToggleJointDiagnosticDisplay()
        {
            _meshExtractor?.ToggleJointDiagnosticDisplay();
        }

        // ─────────────────────────────────────────────────────────────
        //  内部
        // ─────────────────────────────────────────────────────────────

        private void SetupHeadExclusion()
        {
            if (_volumeIntegrator == null) return;

            var cam = Camera.main;
            if (cam != null)
            {
                _volumeIntegrator.ExclusionZones.Add(cam.transform);
                Logger.Info($"头部排除区已添加: {cam.gameObject.name}");
            }
            else
            {
                Logger.Warning("未找到主相机，头部排除区未设置");
            }
        }

        private bool _cameraAvailable;
        private int _colorFrameLog;

        /// <summary>
        /// 每帧把透视相机帧喂给深度双边滤波（RGB 引导）与体积颜色融合。
        /// 与 QRS 原版一致：相机不可用时开启法线回退渲染。
        /// </summary>
        private void ProvideColorFrame()
        {
            ICameraProvider provider = _cameraProvider;

            bool cameraPlaying = provider != null && provider.IsPlaying;

            if (cameraPlaying && !_cameraAvailable)
            {
                _cameraAvailable = true;
                Shader.SetGlobalFloat(NormalFallbackID, 0f);
                Logger.Info("相机已运行 — 关闭法线回退渲染");
            }
            else if (!cameraPlaying && (_cameraAvailable || _colorFrameLog == 0))
            {
                _cameraAvailable = false;
                Shader.SetGlobalFloat(NormalFallbackID, 1f);
                Logger.Info("相机未运行 — 开启法线回退渲染（顶点色将为灰度）");
            }

            if (provider != null && provider.IsReady)
            {
                Texture frame = provider.CurrentFrame;
                if (frame != null)
                {
                    _depthCapture?.SetRGBGuide(frame);

                    Pose pose = provider.CameraPose;
                    if (_depthCapture != null)
                        pose = _depthCapture.TrackingToWorld(pose);

                    _volumeIntegrator.SetCameraData(
                        frame, pose.position, pose.rotation,
                        provider.FocalLength, provider.PrincipalPoint,
                        provider.SensorResolution, provider.CurrentResolution);

                    _colorFrameLog++;
                    if (_colorFrameLog <= 3 || _colorFrameLog % 50 == 0)
                        Logger.Verbose($"彩色帧 #{_colorFrameLog}: {frame.width}x{frame.height}");
                    return;
                }
            }

            _colorFrameLog++;
            _volumeIntegrator.SetCameraData(null, Vector3.zero, Quaternion.identity,
                Vector2.one, Vector2.zero, Vector2.one, Vector2.one);
        }

        private static readonly int NormalFallbackID = Shader.PropertyToID("_RSNormalFallback");
        private static readonly int NoFreezeTintID = Shader.PropertyToID("_RSNoFreezeTint");
        private static readonly int TriAvailableID = Shader.PropertyToID("_RSTriAvailable");
        private static readonly int WireframeID = Shader.PropertyToID("_RSWireframe");
        private static readonly int WireThicknessID = Shader.PropertyToID("_RSWireThickness");

        private void SetSafeShaderDefaults()
        {
            Shader.SetGlobalFloat(TriAvailableID, 0f);
            Shader.SetGlobalFloat(NormalFallbackID, 0f);
            ApplyDisplayMode();
            Shader.SetGlobalFloat(NoFreezeTintID, 0f);
        }

        /// <summary>把当前显示形态写入 shader 全局量（QRS ApplyRenderMode 的线框部分）。</summary>
        private void ApplyDisplayMode()
        {
            Shader.SetGlobalFloat(WireframeID, wireframeMode ? 1f : 0f);
            Shader.SetGlobalFloat(WireThicknessID, wireThickness);
        }
    }
}
