import torch

_lib = torch.mps.compile_shader(r'''
#include <metal_stdlib>
using namespace metal;
kernel void flash(device const half* q, device const half* k, device const half* v,
                  device float* out,
                  uint tg [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half prob[32*32];
    uint head=tg/32, qr=(tg%32)*32+sg*8, base=head*1024*64;
    uint lane=tid%32;
    uint row=(lane%8)/2+(lane/16)*4;
    uint col=(lane%2)*2+((lane/8)%2)*4;
    simdgroup_float8x8 oo[8];
    for(uint i=0;i<8;i++) oo[i]=simdgroup_float8x8(0.f);
    float m=-INFINITY, l=0.f;
    for(uint kb=0;kb<qr+8;kb+=32){
        simdgroup_float8x8 ss[4];
        for(uint i=0;i<4;i++) ss[i]=simdgroup_float8x8(0.f);
        for(uint d=0;d<64;d+=8){
            simdgroup_half8x8 a,b;
            simdgroup_load(a,q+base+qr*64+d,64);
            #pragma unroll
            for(uint i=0;i<4;i++){
                simdgroup_load(b,k+base+(kb+i*8)*64+d,64,ulong2(0),true);
                simdgroup_multiply_accumulate(ss[i],a,b,ss[i]);
            }
        }
        float mx=-INFINITY;
        #pragma unroll
        for(uint i=0;i<4;i++){
            #pragma unroll
            for(uint e=0;e<2;e++){
                float x=kb+i*8+col+e<=qr+row?ss[i].thread_elements()[e]*0.125f:-INFINITY;
                ss[i].thread_elements()[e]=x;
                mx=max(mx,x);
            }
        }
        mx=max(mx,simd_shuffle_xor(mx,1));
        mx=max(mx,simd_shuffle_xor(mx,8));
        mx=max(m,mx);
        float alpha=fast::exp(m-mx), sum=0.f;
        #pragma unroll
        for(uint i=0;i<4;i++){
            simdgroup_half8x8 pp;
            #pragma unroll
            for(uint e=0;e<2;e++){
                float x=fast::exp(ss[i].thread_elements()[e]-mx);
                pp.thread_elements()[e]=half(x);
                sum+=x;
            }
            simdgroup_store(pp,prob+sg*256+i*8,32);
        }
        sum+=simd_shuffle_xor(sum,1);
        sum+=simd_shuffle_xor(sum,8);
        l=l*alpha+sum;
        m=mx;
        #pragma unroll
        for(uint i=0;i<8;i++){
            oo[i].thread_elements()[0]*=alpha;
            oo[i].thread_elements()[1]*=alpha;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for(uint d=0;d<32;d+=8){
            simdgroup_half8x8 a,b;
            simdgroup_load(a,prob+sg*256+d,32);
            #pragma unroll
            for(uint i=0;i<8;i++){
                simdgroup_load(b,v+base+(kb+d)*64+i*8,64);
                simdgroup_multiply_accumulate(oo[i],a,b,oo[i]);
            }
        }
    }
    #pragma unroll
    for(uint i=0;i<8;i++){
        oo[i].thread_elements()[0]/=l;
        oo[i].thread_elements()[1]/=l;
        simdgroup_store(oo[i],out+base+qr*64+i*8,64);
    }
}
''')


def kernel(q, k, v):
    qh, kh, vh = q.half(), k.half(), v.half()
    out = torch.empty_like(q)
    _lib.flash(qh, kh, vh, out, threads=q.numel()//2048*128, group_size=128)
    return out
