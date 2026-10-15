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
    float mx = -INFINITY;
    float4 v0 = logits[row * 1024 + tid + 0];
    mx = max(mx, max(max(v0.x, v0.y), max(v0.z, v0.w)));
    float4 v1 = logits[row * 1024 + tid + 128];
    mx = max(mx, max(max(v1.x, v1.y), max(v1.z, v1.w)));
    float4 v2 = logits[row * 1024 + tid + 256];
    mx = max(mx, max(max(v2.x, v2.y), max(v2.z, v2.w)));
    float4 v3 = logits[row * 1024 + tid + 384];
    mx = max(mx, max(max(v3.x, v3.y), max(v3.z, v3.w)));
    float4 v4 = logits[row * 1024 + tid + 512];
    mx = max(mx, max(max(v4.x, v4.y), max(v4.z, v4.w)));
    float4 v5 = logits[row * 1024 + tid + 640];
    mx = max(mx, max(max(v5.x, v5.y), max(v5.z, v5.w)));
    float4 v6 = logits[row * 1024 + tid + 768];
    mx = max(mx, max(max(v6.x, v6.y), max(v6.z, v6.w)));
    float4 v7 = logits[row * 1024 + tid + 896];
    mx = max(mx, max(max(v7.x, v7.y), max(v7.z, v7.w)));
    threadgroup float scratch[4];
    mx = simd_max(mx);
    if (lane == 0) scratch[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = simd_max(lane < 4 ? scratch[lane] : -INFINITY);
    float sum = 0.0f;
    float4 e0 = exp(v0 - mx);
    sum += (e0.x + e0.y) + (e0.z + e0.w);
    float4 e1 = exp(v1 - mx);
    sum += (e1.x + e1.y) + (e1.z + e1.w);
    float4 e2 = exp(v2 - mx);
    sum += (e2.x + e2.y) + (e2.z + e2.w);
    float4 e3 = exp(v3 - mx);
    sum += (e3.x + e3.y) + (e3.z + e3.w);
    float4 e4 = exp(v4 - mx);
    sum += (e4.x + e4.y) + (e4.z + e4.w);
    float4 e5 = exp(v5 - mx);
    sum += (e5.x + e5.y) + (e5.z + e5.w);
    float4 e6 = exp(v6 - mx);
    sum += (e6.x + e6.y) + (e6.z + e6.w);
    float4 e7 = exp(v7 - mx);
    sum += (e7.x + e7.y) + (e7.z + e7.w);
    sum = simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) scratch[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = simd_sum(lane < 4 ? scratch[lane] : 0.0f);
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
    if logits.shape[1] != 4096 or not logits.is_contiguous() or not y.is_contiguous():
        return (torch.logsumexp(logits, -1) - logits.gather(-1, y[:, None]).squeeze(-1)).mean()
    rows = logits.shape[0]
    losses = torch.empty((rows,), device=logits.device, dtype=torch.float32)
    out = torch.empty((), device=logits.device, dtype=torch.float32)
    _shader.row_loss(logits, y, losses, threads=rows * 128, group_size=128)
    _shader.mean_loss(losses, out, rows, threads=256, group_size=256)
    return out
