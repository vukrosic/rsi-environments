import torch

_lib = torch.mps.compile_shader(r'''
#include <metal_stdlib>
using namespace metal;
kernel void flash(device const half* q, device const half* k, device const half* v,
                  device float* out,
                  uint tg [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half qtile[16*64], kv[64*64];
    threadgroup float score[16*64];
    threadgroup half prob[16*64];
    threadgroup float scales[16];
    threadgroup float maxes[16], sums[16];
    uint head=tg/64, qr=(tg%64)*16;
    uint base=head*1024*64;
    uint row=tid/8, rl=tid%8;
    uint mr=(sg/2)*8, nc=(sg%2)*32;
    uint lane=tid%32, orow=mr+(lane%8)/2+(lane/16)*4;
    simdgroup_float8x8 o0(0.f),o1(0.f),o2(0.f),o3(0.f);
    for(uint idx=tid;idx<16*64;idx+=128) qtile[idx]=q[base+qr*64+idx];
    if(rl==0){maxes[row]=-INFINITY;sums[row]=0.f;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint kb=0;kb<qr+16;kb+=64){
        for(uint idx=tid;idx<64*64;idx+=128) kv[idx]=k[base+kb*64+idx];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 s0(0.f),s1(0.f),s2(0.f),s3(0.f);
        for(uint d=0;d<64;d+=8){
            simdgroup_half8x8 a,b;
            simdgroup_load(a,qtile+mr*64+d,64);
            simdgroup_load(b,kv+nc*64+d,64,ulong2(0),true);
            simdgroup_multiply_accumulate(s0,a,b,s0);
            simdgroup_load(b,kv+(nc+8)*64+d,64,ulong2(0),true);
            simdgroup_multiply_accumulate(s1,a,b,s1);
            simdgroup_load(b,kv+(nc+16)*64+d,64,ulong2(0),true);
            simdgroup_multiply_accumulate(s2,a,b,s2);
            simdgroup_load(b,kv+(nc+24)*64+d,64,ulong2(0),true);
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

        }
        sum+=simd_shuffle_xor(sum,1);
        sum+=simd_shuffle_xor(sum,2);
        sum+=simd_shuffle_xor(sum,4);
        if(rl==0){sums[row]=sums[row]*alpha+sum;maxes[row]=mx;scales[row]=alpha;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint idx=tid;idx<64*64;idx+=128) kv[idx]=v[base+kb*64+idx];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float factor=scales[orow];
        for(uint e=0;e<2;e++){
            o0.thread_elements()[e]*=factor;
            o1.thread_elements()[e]*=factor;
            o2.thread_elements()[e]*=factor;
            o3.thread_elements()[e]*=factor;
        }
        for(uint d=0;d<64;d+=8){
            simdgroup_half8x8 a,b;
            simdgroup_load(a,prob+mr*64+d,64);
            simdgroup_load(b,kv+d*64+nc,64);
            simdgroup_multiply_accumulate(o0,a,b,o0);
            simdgroup_load(b,kv+d*64+nc+8,64);
            simdgroup_multiply_accumulate(o1,a,b,o1);
            simdgroup_load(b,kv+d*64+nc+16,64);
            simdgroup_multiply_accumulate(o2,a,b,o2);
            simdgroup_load(b,kv+d*64+nc+24,64);
            simdgroup_multiply_accumulate(o3,a,b,o3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv=1.f/sums[orow];
    for(uint e=0;e<2;e++){
        o0.thread_elements()[e]*=inv;
        o1.thread_elements()[e]*=inv;
        o2.thread_elements()[e]*=inv;
        o3.thread_elements()[e]*=inv;
    }
    simdgroup_store(o0,out+base+(qr+mr)*64+nc,64);
    simdgroup_store(o1,out+base+(qr+mr)*64+nc+8,64);
    simdgroup_store(o2,out+base+(qr+mr)*64+nc+16,64);
    simdgroup_store(o3,out+base+(qr+mr)*64+nc+24,64);
}
''')


def kernel(q, k, v):
    qh, kh, vh = q.half(), k.half(), v.half()
    out = torch.empty_like(q)
    _lib.flash(qh, kh, vh, out, threads=q.numel()//1024*128, group_size=128)
    return out
