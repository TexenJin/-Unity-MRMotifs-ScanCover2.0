# Phase 1 Depth Surface Adapter P1

## 目标

把 `ScanCoverDepthSurfaceField_P1` 的输入层从“写死的 EnvironmentRaycast 采样”拆成可替换 provider，为后续转向 `DepthGridPointCloud / CustomEnvironmentDepthRaycaster` 路线做准备。

## 新增组件

- `ScanCoverDepthSurfaceProvider_P1`

## 当前支持的后端

1. `EnvironmentRaycast`
- 默认后端
- 复用当前 ScanCover 已有的环境射线采样链
- 作用：保持 Phase 1 当前原型继续可跑

2. `CustomDepthRaycasterReflection`
- 第一版接入后端
- 不直接依赖外部工程程序集
- 通过反射兼容下列方法：
  - `SetEye`
  - `WorldPosAtDepthTexCoord02`
  - `WorldPosToLinearDepth02`
  - `ReconstructNormal02`
- 作用：允许把外部 `CustomEnvironmentDepthRaycaster` 类似能力接到 ScanCover，而不先硬拷贝整套实现

## 当前架构

```text
DepthSurfaceField_P1
  -> SurfaceProvider_P1
     -> EnvironmentRaycast backend
     -> CustomDepthRaycasterReflection backend
```

## 第一版接入范围

已完成：
- provider 抽象组件建立
- `DepthSurfaceField_P1` 优先从 provider 收 observation
- 兼容 observation confidence
- 保留旧 `EnvironmentRaycast` fallback，不破坏当前 Phase 1

未完成：
- 未把外部 `CustomEnvironmentDepthRaycaster` 直接复制进本工程
- 未接入真正的深度纹理生命周期管理
- 未接入密集深度面 patch 壳层生成

## 推荐使用方式

### 当前稳定基线

- `SurfaceProvider_P1.backend = EnvironmentRaycast`
- 继续用现有 Phase 1 样本场 / patch 聚合

### 下一步切换目标

当 ScanCover 工程内存在可用的 depth raycaster 组件后：

- 把 `SurfaceProvider_P1.backend` 切到 `CustomDepthRaycasterReflection`
- 给 `customDepthRaycaster` 绑定目标组件
- 通过 `depthStride / depthSamplesPerStep / depthMinMeters / depthMaxMeters` 控制输入密度

## 工程意义

这一步不是最终 depth-driven surface 本体，只是把“输入替换权”从 `DepthSurfaceField_P1` 中解耦出来。

Phase 1 后续能否真正转向连续 depth surface，取决于：

1. 是否把 `CustomEnvironmentDepthRaycaster` 能力完整迁入 ScanCover
2. 是否从稀疏射线命中升级为更密的深度网格观测
3. 是否基于更密输入重做 patch 壳层与蓝色显示网格
