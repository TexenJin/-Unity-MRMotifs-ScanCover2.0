# Phase 1 Patch Candidate Path

Date: 2026-03-14

## Purpose

Use the dense `world position + world normal + confidence` field from `ScanCoverDepthPreprocessor` to find small local surface patches before attempting any regular lattice or larger surface growth.

## Why This Path Replaced The Fused-Point Mainline

The binocular fused-point branch succeeded as a diagnostic tool, but it did not produce enough stable, dense, local evidence to support a regular ScanCover lattice.

Observed failure mode:

- fused points were sparse
- fused points remained visually irregular
- stable point accumulation still did not provide enough local support to infer a reliable regular node grid

Therefore the point branch is now diagnostic, not primary.

## Current Main Experimental Flow

`EnvironmentDepthManager`
-> `ScanCoverDepthPreprocessor`
-> `ScanCoverSurfacePatchCandidateProvider`
-> `ScanCoverSurfacePatchDebugQuads`

## Patch Candidate Meaning

A patch candidate is not the final ScanCover surface.

It is a small local area that already has enough evidence to be treated as a surface fragment:

- normals are locally coherent
- positions are locally continuous
- confidence is sufficient
- the region is close enough to the camera and not dominated by outliers

Each accepted patch candidate is a small “surface brick” that can later support:

- regular lattice resampling
- patch accumulation
- local surface growth

## Validation Rule

If patch candidates do not appear in coherent clusters on real surfaces such as walls, desks, monitor faces, or cabinet faces, then later lattice generation should not proceed.

The patch layer exists specifically to answer:

“Do we already have enough local surface evidence to grow structured coverage here?”
