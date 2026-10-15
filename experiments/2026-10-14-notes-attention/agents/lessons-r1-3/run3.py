"""Causal attention using tiled Metal matrix operations."""
import torch

_SOURCE = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void attention(
    device const float* Q, device const float* K, device const float* V,
    device float* O,
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint first = group.x * 32;
    const uint base = group.y * 1024 * 64;
    const uint row = (lane % 8) / 2 + (lane / 16) * 4;
    const uint col = (lane % 2) * 2 + ((lane / 8) % 2) * 4;
    const uint qr = sg * 8 + row;
    threadgroup half qs[32 * 64];
    threadgroup half ks[32 * 64];
    threadgroup half vs[32 * 64];
    threadgroup half ps[32 * 32];
    for (uint i=tid; i<32*64; i+=128) qs[i] = half(Q[base + first*64+i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 aq[8];
    simdgroup_float8x8 out[8];
    for (uint d=0; d<8; ++d) {
        simdgroup_load(aq[d], qs+sg*8*64+d*8, 64);
        out[d].thread_elements()[0] = 0;
        out[d].thread_elements()[1] = 0;
    }
    float maximum = -INFINITY;
    float total = 0;
    for (uint start=0; start<first+32; start+=32) {
        for (uint i=tid; i<32*64; i+=128) {
            ks[i] = half(K[base+start*64+i]);
            vs[i] = half(V[base+start*64+i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 score[4];
        for (uint j=0; j<4; ++j) {
            score[j].thread_elements()[0] = 0;
            score[j].thread_elements()[1] = 0;
        }
        for (uint d=0; d<8; ++d) {
            for (uint j=0; j<4; ++j) {
                simdgroup_half8x8 bk;
                simdgroup_load(bk, ks+j*8*64+d*8, 64, ulong2(0), true);
                simdgroup_multiply_accumulate(score[j], aq[d], bk, score[j]);
            }
        }
        float mx = -INFINITY;
        for (uint j=0; j<4; ++j) {
            for (uint e=0; e<2; ++e) {
                float s = score[j].thread_elements()[e] * 0.125f;
                if (start+j*8+col+e > first+qr) s = -INFINITY;
                score[j].thread_elements()[e] = s;
                mx = max(mx, s);
            }
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        float next_max = max(maximum, mx);
        float correction = fast::exp(maximum-next_max);
        float block_sum = 0;
        for (uint j=0; j<4; ++j) {
            simdgroup_half8x8 p;
            for (uint e=0; e<2; ++e) {
                float prob = fast::exp(score[j].thread_elements()[e]-next_max);
                block_sum += prob;
                p.thread_elements()[e] = half(prob);
            }
            simdgroup_store(p, ps+sg*8*32+j*8, 32);
        }
        block_sum += simd_shuffle_xor(block_sum, 1);
        block_sum += simd_shuffle_xor(block_sum, 8);
        total = total*correction + block_sum;
        maximum = next_max;
        for (uint d=0; d<8; ++d) {
            out[d].thread_elements()[0] *= correction;
            out[d].thread_elements()[1] *= correction;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j=0; j<4; ++j) {
            simdgroup_half8x8 p;
            simdgroup_load(p, ps+sg*8*32+j*8, 32);
            for (uint d=0; d<8; ++d) {
                simdgroup_half8x8 b;
                simdgroup_load(b, vs+j*8*64+d*8, 64);
                simdgroup_multiply_accumulate(out[d], p, b, out[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d=0; d<8; ++d) {
        out[d].thread_elements()[0] /= total;
        out[d].thread_elements()[1] /= total;
        simdgroup_store(out[d], O+base+(first+sg*8)*64+d*8, 64);
    }
}
"""
_LIB = torch.mps.compile_shader(_SOURCE)


def kernel(q, k, v):
    if q.shape != (8, 8, 1024, 64):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = torch.empty_like(q)
    _LIB.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(128, 1, 1))
    return out
