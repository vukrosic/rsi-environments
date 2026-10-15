"""Fused, stable cross entropy for contiguous MPS logits."""
import torch

_shader = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;

kernel void row_loss(device const float4* logits,
                     device const uint2* labels,
                     device float* losses,
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    float4 v[8];
    float mx = -INFINITY;
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        v[i] = logits[row * 1024 + tid + i * 128];
        mx = max(mx, max(max(v[i].x, v[i].y), max(v[i].z, v[i].w)));
    }
    mx = simd_max(mx);
    float sum = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        float4 e = exp(v[i] - mx);
        sum += (e.x + e.y) + (e.z + e.w);
    }
    sum = simd_sum(sum);
    threadgroup float2 scratch[4];
    if (lane == 0) scratch[sg] = float2(mx, sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float row_max = max(max(scratch[0].x, scratch[1].x), max(scratch[2].x, scratch[3].x));
        float row_sum = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 4; ++i) row_sum += scratch[i].y * exp(scratch[i].x - row_max);
        mx = row_max;
        sum = row_sum;
        device const float* scalars = reinterpret_cast<device const float*>(logits);
        losses[row] = mx + log(sum) - scalars[row * 4096 + labels[row].x];
    }
}

kernel void mean_loss(device const float* losses,
                      device float* out,
                      constant uint& rows,
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    float sum = 0.0f;
    for (uint i = tid; i < rows; i += 256) sum += losses[i];
    sum = simd_sum(sum);
    threadgroup float scratch[8];
    if (lane == 0) scratch[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        sum = simd_sum(lane < 8 ? scratch[lane] : 0.0f);
        if (lane == 0) out[0] = sum / float(rows);
    }
}
""")


def kernel(logits, y):
    if logits.shape[1] != 4096 or not logits.is_contiguous() or not y.is_contiguous():
        return (torch.logsumexp(logits, -1) - logits.gather(-1, y[:, None]).squeeze(-1)).mean()
    rows = logits.shape[0]
    losses = torch.empty((rows,), device=logits.device, dtype=torch.float32)
    out = torch.empty((), device=logits.device, dtype=torch.float32)
    _shader.row_loss(logits, y, losses, threads=rows * 128, group_size=128)
    _shader.mean_loss(losses, out, rows, threads=256, group_size=256)
    return out
