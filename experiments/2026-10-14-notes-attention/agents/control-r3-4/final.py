"""Metal causal attention specialized for the supplied float32 MPS workload."""
import torch

_BQ = 64
_BK = 32
_QR = 1
_NT = _BQ // (8 * _QR) * 32

_source = f"#define BQ {_BQ}\n#define BK {_BK}\n#define QR {_QR}\n#define NT {_NT}\n#define NC (BK/8)\n" + r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void pack(device const float4* Q [[buffer(0)]],
                 device const float4* K [[buffer(1)]],
                 device const float4* V [[buffer(2)]],
                 device half4* X [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    X[i]=half4(Q[i]);
    X[1048576+i]=half4(K[i]);
    X[2097152+i]=half4(V[i]);
}

kernel void attention(device const half* X [[buffer(0)]],
                      device float* O [[buffer(1)]],
                      uint3 gid [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    device const half* Q=X;
    device const half* K=X+4194304;
    device const half* V=X+8388608;
    threadgroup half qs[BQ*64];
    threadgroup half ks[BK*64];
    threadgroup half vs[BK*64];
    uint base = gid.y*1024*64;
    uint qbase = gid.x*BQ;
    for(uint i=tid;i<BQ*64;i+=NT) qs[i]=Q[base+qbase*64+i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qf[QR][8];
    simdgroup_float8x8 acc[QR][8];
    float maximum[QR], denominator[QR];
    #pragma unroll
    for(uint r=0;r<QR;r++) {
        maximum[r]=0.0f;
        denominator[r]=0.0f;
        #pragma unroll
        for(uint d=0;d<8;d++) {
            simdgroup_load(qf[r][d],qs+(sg*QR+r)*8*64+d*8,64);
            acc[r][d]=make_filled_simdgroup_matrix<float,8>(0.0f);
        }
    }
    // Apple SIMD matrix fragments distribute each row over four lanes.
    uint row0=sg*QR*8+((lane>>1)&3)+(lane>>4)*4;
    uint col0=(lane&1)*2+((lane>>3)&1)*4;
    for(uint start=0;start<qbase+BQ;start+=BK) {
        for(uint i=tid;i<BK*64;i+=NT) {
            ks[i]=K[base+start*64+i];
            vs[i]=V[base+start*64+i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 sf[QR][NC];
        #pragma unroll
        for(uint c=0;c<NC;c++) {
            #pragma unroll
            for(uint r=0;r<QR;r++) sf[r][c]=make_filled_simdgroup_matrix<float,8>(0.0f);
            #pragma unroll
            for(uint d=0;d<8;d++) {
                simdgroup_half8x8 kf;
                simdgroup_load(kf,ks+c*8*64+d*8,64,ulong2(0),true);
                #pragma unroll
                for(uint r=0;r<QR;r++) simdgroup_multiply_accumulate(sf[r][c],qf[r][d],kf,sf[r][c]);
            }
        }
        simdgroup_half8x8 pf[QR][NC];
        float alphas[QR];
        #pragma unroll
        for(uint r=0;r<QR;r++) {
            float vals[NC*2];
            float mx=-INFINITY;
            #pragma unroll
            for(uint c=0;c<NC;c++) {
                #pragma unroll
                for(uint j=0;j<2;j++) {
                    uint col=c*8+col0+j;
                    float s=start+col<=qbase+row0+r*8 ? sf[r][c].thread_elements()[j]*0.18033688011112042f : -INFINITY;
                    vals[c*2+j]=s;
                    mx=max(mx,s);
                }
            }
            float nm=maximum[r];
            float alpha=1.0f;
            // Shift only when needed to keep half matrix operands in range.
            if(simd_any(mx>maximum[r]+12.0f)) {
                mx=max(mx,simd_shuffle_xor(mx,1));
                mx=max(mx,simd_shuffle_xor(mx,8));
                nm=max(maximum[r],mx-12.0f);
                alpha=fast::exp2(maximum[r]-nm);
            }
            float sm=0.0f;
            #pragma unroll
            for(uint c=0;c<NC;c++) {
                #pragma unroll
                for(uint j=0;j<2;j++) {
                    float p=fast::exp2(vals[c*2+j]-nm);
                    pf[r][c].thread_elements()[j]=half(p);
                    sm+=p;
                }
            }
            sm+=simd_shuffle_xor(sm,1);
            sm+=simd_shuffle_xor(sm,8);
            denominator[r]=denominator[r]*alpha+sm;
            maximum[r]=nm;
            alphas[r]=alpha;
        }
        #pragma unroll
        for(uint d=0;d<8;d++) {
            #pragma unroll
            for(uint r=0;r<QR;r++) {
                if(alphas[r]!=1.0f) {
                    acc[r][d].thread_elements()[0]*=alphas[r];
                    acc[r][d].thread_elements()[1]*=alphas[r];
                }
            }
            #pragma unroll
            for(uint c=0;c<NC;c++) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf,vs+c*8*64+d*8,64);
                #pragma unroll
                for(uint r=0;r<QR;r++) simdgroup_multiply_accumulate(acc[r][d],pf[r][c],vf,acc[r][d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint r=0;r<QR;r++) {
        #pragma unroll
        for(uint d=0;d<8;d++) {
            acc[r][d].thread_elements()[0]/=denominator[r];
            acc[r][d].thread_elements()[1]/=denominator[r];
            simdgroup_store(acc[r][d],O+base+(qbase+(sg*QR+r)*8)*64+d*8,64);
        }
    }
}
"""
_lib = torch.mps.compile_shader(_source)


def kernel(q, k, v):
    # One conversion pass; all attention sums and accumulators use float32.
    packed = torch.empty((3, 8, 8, 1024, 64), device=q.device, dtype=torch.float16)
    out = torch.empty_like(q)
    _lib.pack(q, k, v, packed, threads=1048576, group_size=256)
    _lib.attention(packed, out, threads=(1024 // _BQ * _NT, 64, 1), group_size=(_NT, 1, 1))
    return out
