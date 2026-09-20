using System;
using System.Collections.Generic;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Product-level admission gate between an extracted candidate and the
    /// immutable visible front.  It never reads or writes TSDF state.  A bad,
    /// stale or temporarily empty rebuild therefore leaves the last committed
    /// chunk visible instead of turning reconstruction noise into a product.
    /// </summary>
    internal sealed class ChunkQualityGate
    {
        internal enum ProductState
        {
            Empty,
            Candidate,
            Committed,
            Updated,
            Unchanged,
            RemovalPending,
            Removed
        }

        internal readonly struct Decision
        {
            internal readonly bool Publish;
            internal readonly bool Remove;
            internal readonly ProductState State;
            internal readonly string Reason;

            internal Decision(bool publish, bool remove, ProductState state,
                string reason)
            {
                Publish = publish;
                Remove = remove;
                State = state;
                Reason = reason;
            }
        }

        private sealed class State
        {
            internal ProductState ProductState;
            internal uint CommittedEpoch;
            internal int CommittedVertices;
            internal int CommittedIndices;
            internal int RegressionConfirmations;
            internal float RegressionSince;
            internal int RemovalConfirmations;
            internal float RemovalSince;
        }

        private readonly Dictionary<int, State> _states =
            new Dictionary<int, State>();

        private const float MinimumRetainedRatio = 0.70f;
        private const int RegressionConfirmationsRequired = 3;
        private const float RegressionConfirmSeconds = 0.50f;
        private const int RemovalConfirmationsRequired = 5;
        private const float RemovalConfirmSeconds = 1.00f;

        internal Decision Evaluate(int chunkIndex, uint candidateEpoch,
            int vertices, int indices, bool hasCommittedFront,
            uint[] committedOccupancy, uint[] candidateOccupancy)
        {
            State state = GetState(chunkIndex);
            state.ProductState = ProductState.Candidate;

            // An already superseded callback may not move the visible product
            // backwards. A candidate merely followed by a newer dirty epoch is
            // still publishable; rejecting that common case would starve live
            // scanning while the camera keeps observing the room.
            // Equal TSDF epochs are valid: a post-TSDF ruler/court revision can
            // legitimately re-productize the same zero surface.  Request serial
            // and product-revision checks in the pipeline reject truly obsolete
            // asynchronous callbacks before they reach this gate.
            if (hasCommittedFront && candidateEpoch < state.CommittedEpoch)
                return new Decision(false, false, ProductState.Unchanged,
                    "stale_candidate");

            int triangles = Math.Max(0, indices / 3);
            if (vertices <= 0 || triangles <= 0)
            {
                if (!hasCommittedFront)
                    return new Decision(true, true, ProductState.Empty,
                        "empty_initial_candidate");

                float now = Time.realtimeSinceStartup;
                if (state.RemovalConfirmations == 0)
                    state.RemovalSince = now;
                state.RemovalConfirmations++;
                bool confirmed = state.RemovalConfirmations >=
                                     RemovalConfirmationsRequired &&
                                 now - state.RemovalSince >=
                                     RemovalConfirmSeconds;
                state.ProductState = confirmed
                    ? ProductState.Removed
                    : ProductState.RemovalPending;
                return new Decision(confirmed, confirmed, state.ProductState,
                    confirmed ? "confirmed_removal" : "removal_pending");
            }

            state.RemovalConfirmations = 0;
            state.RemovalSince = 0f;

            if (hasCommittedFront && state.CommittedIndices > 0)
            {
                bool vertexRegression = vertices <
                    state.CommittedVertices * MinimumRetainedRatio;
                bool triangleRegression = indices <
                    state.CommittedIndices * MinimumRetainedRatio;
                CountOccupancy(committedOccupancy, candidateOccupancy,
                    out uint occupiedBefore, out uint lostCells);
                bool spatialRegression = occupiedBefore >= 16u &&
                    lostCells > occupiedBefore * (1f - MinimumRetainedRatio);
                if (vertexRegression || triangleRegression || spatialRegression)
                {
                    float now = Time.realtimeSinceStartup;
                    if (state.RegressionConfirmations == 0)
                        state.RegressionSince = now;
                    state.RegressionConfirmations++;
                    bool confirmed = state.RegressionConfirmations >=
                                         RegressionConfirmationsRequired &&
                                     now - state.RegressionSince >=
                                         RegressionConfirmSeconds;
                    if (!confirmed)
                        return new Decision(false, false,
                            ProductState.Unchanged, "coverage_regression_hold");
                }
            }

            state.RegressionConfirmations = 0;
            state.RegressionSince = 0f;
            ProductState acceptedState = hasCommittedFront
                ? ProductState.Updated
                : ProductState.Committed;
            return new Decision(true, false, acceptedState,
                hasCommittedFront ? "quality_update" : "quality_initial");
        }

        internal void RecordPublished(int chunkIndex, uint epoch, int vertices,
            int indices, ProductState productState)
        {
            State state = GetState(chunkIndex);
            state.ProductState = productState;
            state.CommittedEpoch = epoch;
            state.CommittedVertices = Math.Max(0, vertices);
            state.CommittedIndices = Math.Max(0, indices);
            state.RegressionConfirmations = 0;
            state.RemovalConfirmations = 0;
        }

        internal void Clear()
        {
            _states.Clear();
        }

        private State GetState(int chunkIndex)
        {
            if (!_states.TryGetValue(chunkIndex, out State state))
            {
                state = new State { ProductState = ProductState.Empty };
                _states.Add(chunkIndex, state);
            }
            return state;
        }

        private static void CountOccupancy(uint[] committed, uint[] candidate,
            out uint occupiedBefore, out uint lost)
        {
            occupiedBefore = 0u;
            lost = 0u;
            if (committed == null || candidate == null)
                return;
            int count = Math.Min(committed.Length, candidate.Length);
            for (int i = 0; i < count; i++)
            {
                occupiedBefore += PopCount(committed[i]);
                lost += PopCount(committed[i] & ~candidate[i]);
            }
        }

        private static uint PopCount(uint value)
        {
            value -= (value >> 1) & 0x55555555u;
            value = (value & 0x33333333u) +
                    ((value >> 2) & 0x33333333u);
            return (((value + (value >> 4)) & 0x0F0F0F0Fu) *
                    0x01010101u) >> 24;
        }
    }
}
