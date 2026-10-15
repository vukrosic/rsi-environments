import torch

_lib = torch.mps.compile_shader(r'''
#include <metal_stdlib>
using namespace metal;
kernel void flash(device const half* q, device const half* k, device const half* v,
                  device float* out,
                  uint tg [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half prob[32*64];
    threadgroup float group_max[32*2], group_sum[32*2];
    uint head=tg/32, qr=(tg%32)*32, base=head*1024*64;
    uint mr=(sg/2)*8, nc=(sg%2)*32, lane=tid%32;
    uint row=mr+(lane%8)/2+(lane/16)*4;
    uint col=(lane%2)*2+((lane/8)%2)*4;
    bool owner=col==0;
    simdgroup_float8x8 oo[4];
    for(uint i=0;i<4;i++) oo[i]=simdgroup_float8x8(0.f);
    float m=-INFINITY, l=0.f;
    for(uint kb=0;kb<qr+32;kb+=64){
        simdgroup_float8x8 ss[4];
        for(uint i=0;i<4;i++) ss[i]=simdgroup_float8x8(0.f);
        for(uint d=0;d<64;d+=8){
            simdgroup_half8x8 a,b;
            simdgroup_load(a,q+base+(qr+mr)*64+d,64);
            #pragma unroll
            for(uint i=0;i<4;i++){
                simdgroup_load(b,k+base+(kb+nc+i*8)*64+d,64,ulong2(0),true);
                simdgroup_multiply_accumulate(ss[i],a,b,ss[i]);
            }
        }
        float mx=-INFINITY;
        #pragma unroll
        for(uint i=0;i<4;i++){
            #pragma unroll
            for(uint e=0;e<2;e++){
                float x=kb+nc+i*8+col+e<=qr+row?ss[i].thread_elements()[e]*0.125f:-INFINITY;
                ss[i].thread_elements()[e]=x;
                mx=max(mx,x);
            }
        }
        mx=max(mx,simd_shuffle_xor(mx,1));
        mx=max(mx,simd_shuffle_xor(mx,8));
        if(owner) group_max[row*2+sg%2]=mx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mx=max(m,max(group_max[row*2],group_max[row*2+1]));
        float alpha=fast::exp(m-mx), sum=0.f;
        #pragma unroll
        for(uint i=0;i<4;i++){
            simdgroup_half8x8 pp;
            #pragma unroll
            for(uint e=0;e<2;e++){
                float x=fast::exp(ss[i].thread_elements()[e]-mx);
                pp.thread_elements()[e]=half(x);
                sum+=x;
                oo[i].thread_elements()[e]*=alpha;
            }
            simdgroup_store(pp,prob+mr*64+nc+i*8,64);
        }
        sum+=simd_shuffle_xor(sum,1);
        sum+=simd_shuffle_xor(sum,8);
        if(owner) group_sum[row*2+sg%2]=sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        l=l*alpha+group_sum[row*2]+group_sum[row*2+1];
        m=mx;
        for(uint d=0;d<64;d+=8){
            simdgroup_half8x8 a,b;
            simdgroup_load(a,prob+mr*64+d,64);
            #pragma unroll
            for(uint i=0;i<4;i++){
                simdgroup_load(b,v+base+(kb+d)*64+nc+i*8,64);
                simdgroup_multiply_accumulate(oo[i],a,b,oo[i]);
            }
        }
    }
    #pragma unroll
    for(uint i=0;i<4;i++){
        oo[i].thread_elements()[0]/=l;
        oo[i].thread_elements()[1]/=l;
        simdgroup_store(oo[i],out+base+(qr+mr)*64+nc+i*8,64);
    }
}
''')


def kernel(q, k, v):
    qh, kh, vh = q.half(), k.half(), v.half()
    out = torch.empty_like(q)
    _lib.flash(qh, kh, vh, out, threads=q.numel()//2048*256, group_size=256)
    return out
