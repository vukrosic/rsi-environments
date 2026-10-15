import torch

# Each SIMD group handles 16 queries and streams through 64-key tiles.
# Explicit matrix variables keep accumulators in registers on the M4.
_lib = torch.mps.compile_shader(r'''
#include <metal_stdlib>
using namespace metal;
kernel void flash(device const half* q, device const half* k, device const half* v,
                  device float* out,
                  uint tg [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half prob[4096];
    uint head=tg/16, qr=(tg%16)*64+sg*16, base=head*1024*64;
    uint lane=tid%32, row=(lane%8)/2+(lane/16)*4, col=(lane%2)*2+((lane/8)%2)*4;
    float m0=-INFINITY, l0=0.f;
    simdgroup_float8x8 o0_0(0.f);
    simdgroup_float8x8 o0_1(0.f);
    simdgroup_float8x8 o0_2(0.f);
    simdgroup_float8x8 o0_3(0.f);
    simdgroup_float8x8 o0_4(0.f);
    simdgroup_float8x8 o0_5(0.f);
    simdgroup_float8x8 o0_6(0.f);
    simdgroup_float8x8 o0_7(0.f);
    float m1=-INFINITY, l1=0.f;
    simdgroup_float8x8 o1_0(0.f);
    simdgroup_float8x8 o1_1(0.f);
    simdgroup_float8x8 o1_2(0.f);
    simdgroup_float8x8 o1_3(0.f);
    simdgroup_float8x8 o1_4(0.f);
    simdgroup_float8x8 o1_5(0.f);
    simdgroup_float8x8 o1_6(0.f);
    simdgroup_float8x8 o1_7(0.f);
    for(uint kb=0;kb<qr+16;kb+=64){
        simdgroup_float8x8 s0_0(0.f);
        simdgroup_float8x8 s0_1(0.f);
        simdgroup_float8x8 s0_2(0.f);
        simdgroup_float8x8 s0_3(0.f);
        simdgroup_float8x8 s0_4(0.f);
        simdgroup_float8x8 s0_5(0.f);
        simdgroup_float8x8 s0_6(0.f);
        simdgroup_float8x8 s0_7(0.f);
        simdgroup_float8x8 s1_0(0.f);
        simdgroup_float8x8 s1_1(0.f);
        simdgroup_float8x8 s1_2(0.f);
        simdgroup_float8x8 s1_3(0.f);
        simdgroup_float8x8 s1_4(0.f);
        simdgroup_float8x8 s1_5(0.f);
        simdgroup_float8x8 s1_6(0.f);
        simdgroup_float8x8 s1_7(0.f);
        for(uint d=0;d<64;d+=8){
            simdgroup_half8x8 b;
            simdgroup_half8x8 a0;
            simdgroup_load(a0,q+base+(qr+0)*64+d,64);
            simdgroup_half8x8 a1;
            simdgroup_load(a1,q+base+(qr+8)*64+d,64);
            simdgroup_load(b,k+base+(kb+0)*64+d,64,ulong2(0),true);
            simdgroup_multiply_accumulate(s0_0,a0,b,s0_0);
            simdgroup_multiply_accumulate(s1_0,a1,b,s1_0);
            simdgroup_load(b,k+base+(kb+8)*64+d,64,ulong2(0),true);
            simdgroup_multiply_accumulate(s0_1,a0,b,s0_1);
            simdgroup_multiply_accumulate(s1_1,a1,b,s1_1);
            simdgroup_load(b,k+base+(kb+16)*64+d,64,ulong2(0),true);
            simdgroup_multiply_accumulate(s0_2,a0,b,s0_2);
            simdgroup_multiply_accumulate(s1_2,a1,b,s1_2);
            simdgroup_load(b,k+base+(kb+24)*64+d,64,ulong2(0),true);
            simdgroup_multiply_accumulate(s0_3,a0,b,s0_3);
            simdgroup_multiply_accumulate(s1_3,a1,b,s1_3);
            simdgroup_load(b,k+base+(kb+32)*64+d,64,ulong2(0),true);
            simdgroup_multiply_accumulate(s0_4,a0,b,s0_4);
            simdgroup_multiply_accumulate(s1_4,a1,b,s1_4);
            simdgroup_load(b,k+base+(kb+40)*64+d,64,ulong2(0),true);
            simdgroup_multiply_accumulate(s0_5,a0,b,s0_5);
            simdgroup_multiply_accumulate(s1_5,a1,b,s1_5);
            simdgroup_load(b,k+base+(kb+48)*64+d,64,ulong2(0),true);
            simdgroup_multiply_accumulate(s0_6,a0,b,s0_6);
            simdgroup_multiply_accumulate(s1_6,a1,b,s1_6);
            simdgroup_load(b,k+base+(kb+56)*64+d,64,ulong2(0),true);
            simdgroup_multiply_accumulate(s0_7,a0,b,s0_7);
            simdgroup_multiply_accumulate(s1_7,a1,b,s1_7);
        }
        float mx0=-INFINITY;
        s0_0.thread_elements()[0]=kb+0+col+0<=qr+0+row?s0_0.thread_elements()[0]*0.125f:-INFINITY;
        mx0=max(mx0,s0_0.thread_elements()[0]);
        s0_0.thread_elements()[1]=kb+0+col+1<=qr+0+row?s0_0.thread_elements()[1]*0.125f:-INFINITY;
        mx0=max(mx0,s0_0.thread_elements()[1]);
        s0_1.thread_elements()[0]=kb+8+col+0<=qr+0+row?s0_1.thread_elements()[0]*0.125f:-INFINITY;
        mx0=max(mx0,s0_1.thread_elements()[0]);
        s0_1.thread_elements()[1]=kb+8+col+1<=qr+0+row?s0_1.thread_elements()[1]*0.125f:-INFINITY;
        mx0=max(mx0,s0_1.thread_elements()[1]);
        s0_2.thread_elements()[0]=kb+16+col+0<=qr+0+row?s0_2.thread_elements()[0]*0.125f:-INFINITY;
        mx0=max(mx0,s0_2.thread_elements()[0]);
        s0_2.thread_elements()[1]=kb+16+col+1<=qr+0+row?s0_2.thread_elements()[1]*0.125f:-INFINITY;
        mx0=max(mx0,s0_2.thread_elements()[1]);
        s0_3.thread_elements()[0]=kb+24+col+0<=qr+0+row?s0_3.thread_elements()[0]*0.125f:-INFINITY;
        mx0=max(mx0,s0_3.thread_elements()[0]);
        s0_3.thread_elements()[1]=kb+24+col+1<=qr+0+row?s0_3.thread_elements()[1]*0.125f:-INFINITY;
        mx0=max(mx0,s0_3.thread_elements()[1]);
        s0_4.thread_elements()[0]=kb+32+col+0<=qr+0+row?s0_4.thread_elements()[0]*0.125f:-INFINITY;
        mx0=max(mx0,s0_4.thread_elements()[0]);
        s0_4.thread_elements()[1]=kb+32+col+1<=qr+0+row?s0_4.thread_elements()[1]*0.125f:-INFINITY;
        mx0=max(mx0,s0_4.thread_elements()[1]);
        s0_5.thread_elements()[0]=kb+40+col+0<=qr+0+row?s0_5.thread_elements()[0]*0.125f:-INFINITY;
        mx0=max(mx0,s0_5.thread_elements()[0]);
        s0_5.thread_elements()[1]=kb+40+col+1<=qr+0+row?s0_5.thread_elements()[1]*0.125f:-INFINITY;
        mx0=max(mx0,s0_5.thread_elements()[1]);
        s0_6.thread_elements()[0]=kb+48+col+0<=qr+0+row?s0_6.thread_elements()[0]*0.125f:-INFINITY;
        mx0=max(mx0,s0_6.thread_elements()[0]);
        s0_6.thread_elements()[1]=kb+48+col+1<=qr+0+row?s0_6.thread_elements()[1]*0.125f:-INFINITY;
        mx0=max(mx0,s0_6.thread_elements()[1]);
        s0_7.thread_elements()[0]=kb+56+col+0<=qr+0+row?s0_7.thread_elements()[0]*0.125f:-INFINITY;
        mx0=max(mx0,s0_7.thread_elements()[0]);
        s0_7.thread_elements()[1]=kb+56+col+1<=qr+0+row?s0_7.thread_elements()[1]*0.125f:-INFINITY;
        mx0=max(mx0,s0_7.thread_elements()[1]);
        mx0=max(mx0,simd_shuffle_xor(mx0,1));
        mx0=max(mx0,simd_shuffle_xor(mx0,8));
        mx0=max(m0,mx0);
        float alpha0=fast::exp(m0-mx0), sum0=0.f;
        simdgroup_half8x8 p0_0;
        {float x=fast::exp(s0_0.thread_elements()[0]-mx0);p0_0.thread_elements()[0]=half(x);sum0+=x;}
        {float x=fast::exp(s0_0.thread_elements()[1]-mx0);p0_0.thread_elements()[1]=half(x);sum0+=x;}
        simdgroup_store(p0_0,prob+sg*1024+0,64);
        simdgroup_half8x8 p0_1;
        {float x=fast::exp(s0_1.thread_elements()[0]-mx0);p0_1.thread_elements()[0]=half(x);sum0+=x;}
        {float x=fast::exp(s0_1.thread_elements()[1]-mx0);p0_1.thread_elements()[1]=half(x);sum0+=x;}
        simdgroup_store(p0_1,prob+sg*1024+8,64);
        simdgroup_half8x8 p0_2;
        {float x=fast::exp(s0_2.thread_elements()[0]-mx0);p0_2.thread_elements()[0]=half(x);sum0+=x;}
        {float x=fast::exp(s0_2.thread_elements()[1]-mx0);p0_2.thread_elements()[1]=half(x);sum0+=x;}
        simdgroup_store(p0_2,prob+sg*1024+16,64);
        simdgroup_half8x8 p0_3;
        {float x=fast::exp(s0_3.thread_elements()[0]-mx0);p0_3.thread_elements()[0]=half(x);sum0+=x;}
        {float x=fast::exp(s0_3.thread_elements()[1]-mx0);p0_3.thread_elements()[1]=half(x);sum0+=x;}
        simdgroup_store(p0_3,prob+sg*1024+24,64);
        simdgroup_half8x8 p0_4;
        {float x=fast::exp(s0_4.thread_elements()[0]-mx0);p0_4.thread_elements()[0]=half(x);sum0+=x;}
        {float x=fast::exp(s0_4.thread_elements()[1]-mx0);p0_4.thread_elements()[1]=half(x);sum0+=x;}
        simdgroup_store(p0_4,prob+sg*1024+32,64);
        simdgroup_half8x8 p0_5;
        {float x=fast::exp(s0_5.thread_elements()[0]-mx0);p0_5.thread_elements()[0]=half(x);sum0+=x;}
        {float x=fast::exp(s0_5.thread_elements()[1]-mx0);p0_5.thread_elements()[1]=half(x);sum0+=x;}
        simdgroup_store(p0_5,prob+sg*1024+40,64);
        simdgroup_half8x8 p0_6;
        {float x=fast::exp(s0_6.thread_elements()[0]-mx0);p0_6.thread_elements()[0]=half(x);sum0+=x;}
        {float x=fast::exp(s0_6.thread_elements()[1]-mx0);p0_6.thread_elements()[1]=half(x);sum0+=x;}
        simdgroup_store(p0_6,prob+sg*1024+48,64);
        simdgroup_half8x8 p0_7;
        {float x=fast::exp(s0_7.thread_elements()[0]-mx0);p0_7.thread_elements()[0]=half(x);sum0+=x;}
        {float x=fast::exp(s0_7.thread_elements()[1]-mx0);p0_7.thread_elements()[1]=half(x);sum0+=x;}
        simdgroup_store(p0_7,prob+sg*1024+56,64);
        sum0+=simd_shuffle_xor(sum0,1);
        sum0+=simd_shuffle_xor(sum0,8);
        l0=l0*alpha0+sum0; m0=mx0;
        o0_0.thread_elements()[0]*=alpha0;
        o0_0.thread_elements()[1]*=alpha0;
        o0_1.thread_elements()[0]*=alpha0;
        o0_1.thread_elements()[1]*=alpha0;
        o0_2.thread_elements()[0]*=alpha0;
        o0_2.thread_elements()[1]*=alpha0;
        o0_3.thread_elements()[0]*=alpha0;
        o0_3.thread_elements()[1]*=alpha0;
        o0_4.thread_elements()[0]*=alpha0;
        o0_4.thread_elements()[1]*=alpha0;
        o0_5.thread_elements()[0]*=alpha0;
        o0_5.thread_elements()[1]*=alpha0;
        o0_6.thread_elements()[0]*=alpha0;
        o0_6.thread_elements()[1]*=alpha0;
        o0_7.thread_elements()[0]*=alpha0;
        o0_7.thread_elements()[1]*=alpha0;
        float mx1=-INFINITY;
        s1_0.thread_elements()[0]=kb+0+col+0<=qr+8+row?s1_0.thread_elements()[0]*0.125f:-INFINITY;
        mx1=max(mx1,s1_0.thread_elements()[0]);
        s1_0.thread_elements()[1]=kb+0+col+1<=qr+8+row?s1_0.thread_elements()[1]*0.125f:-INFINITY;
        mx1=max(mx1,s1_0.thread_elements()[1]);
        s1_1.thread_elements()[0]=kb+8+col+0<=qr+8+row?s1_1.thread_elements()[0]*0.125f:-INFINITY;
        mx1=max(mx1,s1_1.thread_elements()[0]);
        s1_1.thread_elements()[1]=kb+8+col+1<=qr+8+row?s1_1.thread_elements()[1]*0.125f:-INFINITY;
        mx1=max(mx1,s1_1.thread_elements()[1]);
        s1_2.thread_elements()[0]=kb+16+col+0<=qr+8+row?s1_2.thread_elements()[0]*0.125f:-INFINITY;
        mx1=max(mx1,s1_2.thread_elements()[0]);
        s1_2.thread_elements()[1]=kb+16+col+1<=qr+8+row?s1_2.thread_elements()[1]*0.125f:-INFINITY;
        mx1=max(mx1,s1_2.thread_elements()[1]);
        s1_3.thread_elements()[0]=kb+24+col+0<=qr+8+row?s1_3.thread_elements()[0]*0.125f:-INFINITY;
        mx1=max(mx1,s1_3.thread_elements()[0]);
        s1_3.thread_elements()[1]=kb+24+col+1<=qr+8+row?s1_3.thread_elements()[1]*0.125f:-INFINITY;
        mx1=max(mx1,s1_3.thread_elements()[1]);
        s1_4.thread_elements()[0]=kb+32+col+0<=qr+8+row?s1_4.thread_elements()[0]*0.125f:-INFINITY;
        mx1=max(mx1,s1_4.thread_elements()[0]);
        s1_4.thread_elements()[1]=kb+32+col+1<=qr+8+row?s1_4.thread_elements()[1]*0.125f:-INFINITY;
        mx1=max(mx1,s1_4.thread_elements()[1]);
        s1_5.thread_elements()[0]=kb+40+col+0<=qr+8+row?s1_5.thread_elements()[0]*0.125f:-INFINITY;
        mx1=max(mx1,s1_5.thread_elements()[0]);
        s1_5.thread_elements()[1]=kb+40+col+1<=qr+8+row?s1_5.thread_elements()[1]*0.125f:-INFINITY;
        mx1=max(mx1,s1_5.thread_elements()[1]);
        s1_6.thread_elements()[0]=kb+48+col+0<=qr+8+row?s1_6.thread_elements()[0]*0.125f:-INFINITY;
        mx1=max(mx1,s1_6.thread_elements()[0]);
        s1_6.thread_elements()[1]=kb+48+col+1<=qr+8+row?s1_6.thread_elements()[1]*0.125f:-INFINITY;
        mx1=max(mx1,s1_6.thread_elements()[1]);
        s1_7.thread_elements()[0]=kb+56+col+0<=qr+8+row?s1_7.thread_elements()[0]*0.125f:-INFINITY;
        mx1=max(mx1,s1_7.thread_elements()[0]);
        s1_7.thread_elements()[1]=kb+56+col+1<=qr+8+row?s1_7.thread_elements()[1]*0.125f:-INFINITY;
        mx1=max(mx1,s1_7.thread_elements()[1]);
        mx1=max(mx1,simd_shuffle_xor(mx1,1));
        mx1=max(mx1,simd_shuffle_xor(mx1,8));
        mx1=max(m1,mx1);
        float alpha1=fast::exp(m1-mx1), sum1=0.f;
        simdgroup_half8x8 p1_0;
        {float x=fast::exp(s1_0.thread_elements()[0]-mx1);p1_0.thread_elements()[0]=half(x);sum1+=x;}
        {float x=fast::exp(s1_0.thread_elements()[1]-mx1);p1_0.thread_elements()[1]=half(x);sum1+=x;}
        simdgroup_store(p1_0,prob+sg*1024+512,64);
        simdgroup_half8x8 p1_1;
        {float x=fast::exp(s1_1.thread_elements()[0]-mx1);p1_1.thread_elements()[0]=half(x);sum1+=x;}
        {float x=fast::exp(s1_1.thread_elements()[1]-mx1);p1_1.thread_elements()[1]=half(x);sum1+=x;}
        simdgroup_store(p1_1,prob+sg*1024+520,64);
        simdgroup_half8x8 p1_2;
        {float x=fast::exp(s1_2.thread_elements()[0]-mx1);p1_2.thread_elements()[0]=half(x);sum1+=x;}
        {float x=fast::exp(s1_2.thread_elements()[1]-mx1);p1_2.thread_elements()[1]=half(x);sum1+=x;}
        simdgroup_store(p1_2,prob+sg*1024+528,64);
        simdgroup_half8x8 p1_3;
        {float x=fast::exp(s1_3.thread_elements()[0]-mx1);p1_3.thread_elements()[0]=half(x);sum1+=x;}
        {float x=fast::exp(s1_3.thread_elements()[1]-mx1);p1_3.thread_elements()[1]=half(x);sum1+=x;}
        simdgroup_store(p1_3,prob+sg*1024+536,64);
        simdgroup_half8x8 p1_4;
        {float x=fast::exp(s1_4.thread_elements()[0]-mx1);p1_4.thread_elements()[0]=half(x);sum1+=x;}
        {float x=fast::exp(s1_4.thread_elements()[1]-mx1);p1_4.thread_elements()[1]=half(x);sum1+=x;}
        simdgroup_store(p1_4,prob+sg*1024+544,64);
        simdgroup_half8x8 p1_5;
        {float x=fast::exp(s1_5.thread_elements()[0]-mx1);p1_5.thread_elements()[0]=half(x);sum1+=x;}
        {float x=fast::exp(s1_5.thread_elements()[1]-mx1);p1_5.thread_elements()[1]=half(x);sum1+=x;}
        simdgroup_store(p1_5,prob+sg*1024+552,64);
        simdgroup_half8x8 p1_6;
        {float x=fast::exp(s1_6.thread_elements()[0]-mx1);p1_6.thread_elements()[0]=half(x);sum1+=x;}
        {float x=fast::exp(s1_6.thread_elements()[1]-mx1);p1_6.thread_elements()[1]=half(x);sum1+=x;}
        simdgroup_store(p1_6,prob+sg*1024+560,64);
        simdgroup_half8x8 p1_7;
        {float x=fast::exp(s1_7.thread_elements()[0]-mx1);p1_7.thread_elements()[0]=half(x);sum1+=x;}
        {float x=fast::exp(s1_7.thread_elements()[1]-mx1);p1_7.thread_elements()[1]=half(x);sum1+=x;}
        simdgroup_store(p1_7,prob+sg*1024+568,64);
        sum1+=simd_shuffle_xor(sum1,1);
        sum1+=simd_shuffle_xor(sum1,8);
        l1=l1*alpha1+sum1; m1=mx1;
        o1_0.thread_elements()[0]*=alpha1;
        o1_0.thread_elements()[1]*=alpha1;
        o1_1.thread_elements()[0]*=alpha1;
        o1_1.thread_elements()[1]*=alpha1;
        o1_2.thread_elements()[0]*=alpha1;
        o1_2.thread_elements()[1]*=alpha1;
        o1_3.thread_elements()[0]*=alpha1;
        o1_3.thread_elements()[1]*=alpha1;
        o1_4.thread_elements()[0]*=alpha1;
        o1_4.thread_elements()[1]*=alpha1;
        o1_5.thread_elements()[0]*=alpha1;
        o1_5.thread_elements()[1]*=alpha1;
        o1_6.thread_elements()[0]*=alpha1;
        o1_6.thread_elements()[1]*=alpha1;
        o1_7.thread_elements()[0]*=alpha1;
        o1_7.thread_elements()[1]*=alpha1;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for(uint d=0;d<64;d+=8){
            simdgroup_half8x8 b;
            simdgroup_half8x8 a0;
            simdgroup_load(a0,prob+sg*1024+0+d,64);
            simdgroup_half8x8 a1;
            simdgroup_load(a1,prob+sg*1024+512+d,64);
            simdgroup_load(b,v+base+(kb+d)*64+0,64);
            simdgroup_multiply_accumulate(o0_0,a0,b,o0_0);
            simdgroup_multiply_accumulate(o1_0,a1,b,o1_0);
            simdgroup_load(b,v+base+(kb+d)*64+8,64);
            simdgroup_multiply_accumulate(o0_1,a0,b,o0_1);
            simdgroup_multiply_accumulate(o1_1,a1,b,o1_1);
            simdgroup_load(b,v+base+(kb+d)*64+16,64);
            simdgroup_multiply_accumulate(o0_2,a0,b,o0_2);
            simdgroup_multiply_accumulate(o1_2,a1,b,o1_2);
            simdgroup_load(b,v+base+(kb+d)*64+24,64);
            simdgroup_multiply_accumulate(o0_3,a0,b,o0_3);
            simdgroup_multiply_accumulate(o1_3,a1,b,o1_3);
            simdgroup_load(b,v+base+(kb+d)*64+32,64);
            simdgroup_multiply_accumulate(o0_4,a0,b,o0_4);
            simdgroup_multiply_accumulate(o1_4,a1,b,o1_4);
            simdgroup_load(b,v+base+(kb+d)*64+40,64);
            simdgroup_multiply_accumulate(o0_5,a0,b,o0_5);
            simdgroup_multiply_accumulate(o1_5,a1,b,o1_5);
            simdgroup_load(b,v+base+(kb+d)*64+48,64);
            simdgroup_multiply_accumulate(o0_6,a0,b,o0_6);
            simdgroup_multiply_accumulate(o1_6,a1,b,o1_6);
            simdgroup_load(b,v+base+(kb+d)*64+56,64);
            simdgroup_multiply_accumulate(o0_7,a0,b,o0_7);
            simdgroup_multiply_accumulate(o1_7,a1,b,o1_7);
        }
    }
    o0_0.thread_elements()[0]/=l0;
    o0_0.thread_elements()[1]/=l0;
    simdgroup_store(o0_0,out+base+(qr+0)*64+0,64);
    o0_1.thread_elements()[0]/=l0;
    o0_1.thread_elements()[1]/=l0;
    simdgroup_store(o0_1,out+base+(qr+0)*64+8,64);
    o0_2.thread_elements()[0]/=l0;
    o0_2.thread_elements()[1]/=l0;
    simdgroup_store(o0_2,out+base+(qr+0)*64+16,64);
    o0_3.thread_elements()[0]/=l0;
    o0_3.thread_elements()[1]/=l0;
    simdgroup_store(o0_3,out+base+(qr+0)*64+24,64);
    o0_4.thread_elements()[0]/=l0;
    o0_4.thread_elements()[1]/=l0;
    simdgroup_store(o0_4,out+base+(qr+0)*64+32,64);
    o0_5.thread_elements()[0]/=l0;
    o0_5.thread_elements()[1]/=l0;
    simdgroup_store(o0_5,out+base+(qr+0)*64+40,64);
    o0_6.thread_elements()[0]/=l0;
    o0_6.thread_elements()[1]/=l0;
    simdgroup_store(o0_6,out+base+(qr+0)*64+48,64);
    o0_7.thread_elements()[0]/=l0;
    o0_7.thread_elements()[1]/=l0;
    simdgroup_store(o0_7,out+base+(qr+0)*64+56,64);
    o1_0.thread_elements()[0]/=l1;
    o1_0.thread_elements()[1]/=l1;
    simdgroup_store(o1_0,out+base+(qr+8)*64+0,64);
    o1_1.thread_elements()[0]/=l1;
    o1_1.thread_elements()[1]/=l1;
    simdgroup_store(o1_1,out+base+(qr+8)*64+8,64);
    o1_2.thread_elements()[0]/=l1;
    o1_2.thread_elements()[1]/=l1;
    simdgroup_store(o1_2,out+base+(qr+8)*64+16,64);
    o1_3.thread_elements()[0]/=l1;
    o1_3.thread_elements()[1]/=l1;
    simdgroup_store(o1_3,out+base+(qr+8)*64+24,64);
    o1_4.thread_elements()[0]/=l1;
    o1_4.thread_elements()[1]/=l1;
    simdgroup_store(o1_4,out+base+(qr+8)*64+32,64);
    o1_5.thread_elements()[0]/=l1;
    o1_5.thread_elements()[1]/=l1;
    simdgroup_store(o1_5,out+base+(qr+8)*64+40,64);
    o1_6.thread_elements()[0]/=l1;
    o1_6.thread_elements()[1]/=l1;
    simdgroup_store(o1_6,out+base+(qr+8)*64+48,64);
    o1_7.thread_elements()[0]/=l1;
    o1_7.thread_elements()[1]/=l1;
    simdgroup_store(o1_7,out+base+(qr+8)*64+56,64);
}
''')


def kernel(q, k, v):
    # Half operands, float accumulators and normalization, float32 GPU output.
    qh, kh, vh = q.half(), k.half(), v.half()
    out = torch.empty_like(q)
    _lib.flash(qh, kh, vh, out, threads=q.numel()//4096*128, group_size=128)
    return out
