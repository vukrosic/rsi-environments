import torch

_lib = torch.mps.compile_shader(r'''
#include <metal_stdlib>
using namespace metal;
kernel void causal_softmax(device const half* s, device half* p,
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    uint last = row % 1024;
    float x[4];
    float mx = -INFINITY;
    for (uint i=0;i<4;i++) {
        uint col=tid+i*256;
        x[i] = col<=last ? float(s[row*1024+col])*0.125f : -INFINITY;
        mx=max(mx,x[i]);
    }
    mx=simd_max(mx);
    if(lane==0) scratch[sg]=mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx=simd_max(lane<8?scratch[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum=0;
    for(uint i=0;i<4;i++){x[i]=fast::exp(x[i]-mx);sum+=x[i];}
    sum=simd_sum(sum);
    if(lane==0) scratch[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=simd_sum(lane<8?scratch[lane]:0.f);
    for(uint i=0;i<4;i++) p[row*1024+tid+i*256]=half(x[i]/sum);
}
kernel void flash(device const half* q, device const half* k, device const half* v,
                  device float* out,
                  uint tg [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float score[32*64];
    threadgroup half prob[32*64];
    threadgroup float acc[32*64];
    threadgroup float maxes[32], sums[32];
    uint head=tg/32, qr=(tg%32)*32;
    uint base=head*1024*64;
    uint row=tid/8, rl=tid%8;
    for(uint c=rl;c<64;c+=8) acc[row*64+c]=0.f;
    if(rl==0){maxes[row]=-INFINITY;sums[row]=0.f;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint kb=0;kb<qr+32;kb+=64){
        uint mr=(sg/2)*8, nc=(sg%2)*32;
        simdgroup_float8x8 s0(0.f),s1(0.f),s2(0.f),s3(0.f);
        for(uint d=0;d<64;d+=8){
            simdgroup_half8x8 a,b;
            simdgroup_load(a,q+base+(qr+mr)*64+d,64);
            simdgroup_load(b,k+base+(kb+nc)*64+d,64,ulong2(0),true);
            simdgroup_multiply_accumulate(s0,a,b,s0);
            simdgroup_load(b,k+base+(kb+nc+8)*64+d,64,ulong2(0),true);
            simdgroup_multiply_accumulate(s1,a,b,s1);
            simdgroup_load(b,k+base+(kb+nc+16)*64+d,64,ulong2(0),true);
            simdgroup_multiply_accumulate(s2,a,b,s2);
            simdgroup_load(b,k+base+(kb+nc+24)*64+d,64,ulong2(0),true);
            simdgroup_multiply_accumulate(s3,a,b,s3);
        }
        simdgroup_store(s0,score+mr*64+nc,64);
        simdgroup_store(s1,score+mr*64+nc+8,64);
        simdgroup_store(s2,score+mr*64+nc+16,64);
        simdgroup_store(s3,score+mr*64+nc+24,64);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float xs[8]; float mx=-INFINITY;
        for(uint i=0;i<8;i++){
            uint c=rl+i*8;
            xs[i]=kb+c<=qr+row?score[row*64+c]*0.125f:-INFINITY;
            mx=max(mx,xs[i]);
        }
        mx=max(mx,simd_shuffle_xor(mx,1));
        mx=max(mx,simd_shuffle_xor(mx,2));
        mx=max(mx,simd_shuffle_xor(mx,4));
        mx=max(mx,maxes[row]);
        float alpha=fast::exp(maxes[row]-mx), sum=0.f;
        for(uint i=0;i<8;i++){
            xs[i]=fast::exp(xs[i]-mx);
            sum+=xs[i];
            prob[row*64+rl+i*8]=half(xs[i]);
            acc[row*64+rl+i*8]*=alpha;
        }
        sum+=simd_shuffle_xor(sum,1);
        sum+=simd_shuffle_xor(sum,2);
        sum+=simd_shuffle_xor(sum,4);
        if(rl==0){sums[row]=sums[row]*alpha+sum;maxes[row]=mx;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 o0,o1,o2,o3;
        simdgroup_load(o0,acc+mr*64+nc,64);
        simdgroup_load(o1,acc+mr*64+nc+8,64);
        simdgroup_load(o2,acc+mr*64+nc+16,64);
        simdgroup_load(o3,acc+mr*64+nc+24,64);
        for(uint d=0;d<64;d+=8){
            simdgroup_half8x8 a,b;
            simdgroup_load(a,prob+mr*64+d,64);
            simdgroup_load(b,v+base+(kb+d)*64+nc,64);
            simdgroup_multiply_accumulate(o0,a,b,o0);
            simdgroup_load(b,v+base+(kb+d)*64+nc+8,64);
            simdgroup_multiply_accumulate(o1,a,b,o1);
            simdgroup_load(b,v+base+(kb+d)*64+nc+16,64);
            simdgroup_multiply_accumulate(o2,a,b,o2);
            simdgroup_load(b,v+base+(kb+d)*64+nc+24,64);
            simdgroup_multiply_accumulate(o3,a,b,o3);
        }
        simdgroup_store(o0,acc+mr*64+nc,64);
        simdgroup_store(o1,acc+mr*64+nc+8,64);
        simdgroup_store(o2,acc+mr*64+nc+16,64);
        simdgroup_store(o3,acc+mr*64+nc+24,64);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint c=rl;c<64;c+=8) out[base+(qr+row)*64+c]=acc[row*64+c]/sums[row];
}
''')


def kernel(q, k, v):
    qh, kh, vh = q.half(), k.half(), v.half()
    out = torch.empty_like(q)
    _lib.flash(qh, kh, vh, out, threads=q.numel()//2048*256, group_size=256)
    return out
