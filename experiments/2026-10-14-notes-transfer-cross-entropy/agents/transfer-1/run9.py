"""Fused, stable cross entropy for contiguous MPS logits."""
import torch

_shader = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;

kernel void row_loss(device const float4* logits,
                     device const uint2* labels,
                     device float* losses,
                     uint group [[threadgroup_position_in_grid]],
                     uint thread_id [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = group * 2 + sg / 4;
    uint tid = thread_id % 128;
    uint row_slot = (sg / 4) * 4;
    float4 v[8];
    float mx = -INFINITY;
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        v[i] = logits[row * 1024 + tid + i * 128];
        mx = max(mx, max(max(v[i].x, v[i].y), max(v[i].z, v[i].w)));
    }
    threadgroup float scratch[8];
    mx = simd_max(mx);
    if (lane == 0) scratch[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = simd_max(lane < 4 ? scratch[row_slot + lane] : -INFINITY);
    float sum = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        float4 e = exp(v[i] - mx);
        sum += (e.x + e.y) + (e.z + e.w);
    }
    sum = simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) scratch[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = simd_sum(lane < 4 ? scratch[row_slot + lane] : 0.0f);
    if (tid == 0) {
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
    if logits.shape[0] % 2 or logits.shape[1] != 4096 or not logits.is_contiguous() or not y.is_contiguous():
        return (torch.logsumexp(logits, -1) - logits.gather(-1, y[:, None]).squeeze(-1)).mean()
    rows = logits.shape[0]
    losses = torch.empty((rows,), device=logits.device, dtype=torch.float32)
    out = torch.empty((), device=logits.device, dtype=torch.float32)
    _shader.row_loss(logits, y, losses, threads=rows * 128, group_size=256)
    _shader.mean_loss(losses, out, rows, threads=256, group_size=256)
    return out
