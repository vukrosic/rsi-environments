import torch

# M4 SIMD matrix products use half storage with float accumulation.
# Symmetric products compute one triangle and mirror the result.
_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
template<uint K,uint N,bool POLY,bool TRANS>
void save(simdgroup_float8x8 v,device const half* a,device half* out,
    uint row,uint col,uint lane,bool mirror) {
    uint r=row+(lane/16)*4+(lane/2)%4;
    uint c=col+((lane/8)%2)*4+(lane%2)*2;
    for(uint e=0;e<2;e++) {
        float z=v.thread_elements()[e];
        if(POLY) z=2.0315f*z-4.775f*float(a[r*K+c+e])+(r==c+e?3.4445f:0.0f);
        half h=half(z);
        out[r*N+c+e]=h;
        if((TRANS || POLY) && mirror) out[(c+e)*N+r]=h;
    }
}
template<uint K, uint N, bool TRANS, bool POLY, bool SYM=false>
void mm(device const half* a, device const half* b, device half* out,
        uint3 group, uint sg, uint lane) {
    uint batch=group.z, row=group.x*32+(sg/2)*16, col=group.y*32+(sg%2)*16;
    a+=batch*256*K;
    b+=batch*(TRANS?N*K:K*N);
    out+=batch*256*N;
    simdgroup_float8x8 c00(0),c01(0),c10(0),c11(0);
    for(uint k=0;k<K;k+=8) {
        simdgroup_half8x8 a0,a1,b0,b1;
        simdgroup_load(a0,a+row*K+k,K);
        simdgroup_load(a1,a+(row+8)*K+k,K);
        if(TRANS) {
            simdgroup_load(b0,b+col*K+k,K,ulong2(0),true);
            simdgroup_load(b1,b+(col+8)*K+k,K,ulong2(0),true);
        } else {
            simdgroup_load(b0,b+k*N+col,N);
            simdgroup_load(b1,b+k*N+col+8,N);
        }
        simdgroup_multiply_accumulate(c00,a0,b0,c00);
        simdgroup_multiply_accumulate(c01,a0,b1,c01);
        simdgroup_multiply_accumulate(c10,a1,b0,c10);
        simdgroup_multiply_accumulate(c11,a1,b1,c11);
    }
    save<K,N,POLY,TRANS || SYM>(c00,a,out,row,col,lane,group.x!=group.y);
    save<K,N,POLY,TRANS || SYM>(c01,a,out,row,col+8,lane,group.x!=group.y);
    save<K,N,POLY,TRANS || SYM>(c10,a,out,row+8,col,lane,group.x!=group.y);
    save<K,N,POLY,TRANS || SYM>(c11,a,out,row+8,col+8,lane,group.x!=group.y);
}
kernel void gram(device const half* x,device half* a,uint3 g [[threadgroup_position_in_grid]],
 uint sg [[simdgroup_index_in_threadgroup]],uint l [[thread_index_in_simdgroup]]) {
    if(g.y<g.x) return;
    mm<704,256,true,false>(x,x,a,g,sg,l);
}
kernel void square(device const half* a,device half* b,uint3 g [[threadgroup_position_in_grid]],
 uint sg [[simdgroup_index_in_threadgroup]],uint l [[thread_index_in_simdgroup]]) {
    if(g.y<g.x) return;
    mm<256,256,false,true>(a,a,b,g,sg,l);
}
kernel void smallsym(device const half* a,device const half* b,device half* c,
 uint3 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint l [[thread_index_in_simdgroup]]) {
    if(g.y<g.x) return;
    mm<256,256,false,false,true>(a,b,c,g,sg,l);
}
kernel void dual(device const half* p,device const half* a,device const half* q,
 device half* c,device half* r,uint3 g [[threadgroup_position_in_grid]],
 uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    if(g.y<g.x) return;
    uint row=g.x*32+(sg/2)*16,col=g.y*32+(sg%2)*16;
    p+=g.z*65536; a+=g.z*65536; q+=g.z*65536;
    c+=g.z*65536; r+=g.z*65536;
    simdgroup_float8x8 cc[4],rr[4];
    #pragma unroll
    for(uint i=0;i<4;i++) { cc[i]=simdgroup_float8x8(0);rr[i]=simdgroup_float8x8(0); }
    for(uint k=0;k<256;k+=8) {
        simdgroup_half8x8 pp[2],aa[2],qq[2];
        #pragma unroll
        for(uint i=0;i<2;i++) {
            simdgroup_load(pp[i],p+(row+i*8)*256+k,256);
            simdgroup_load(aa[i],a+k*256+col+i*8,256);
            simdgroup_load(qq[i],q+k*256+col+i*8,256);
        }
        #pragma unroll
        for(uint i=0;i<2;i++) {
            #pragma unroll
            for(uint j=0;j<2;j++) {
                simdgroup_multiply_accumulate(cc[i*2+j],pp[i],aa[j],cc[i*2+j]);
                simdgroup_multiply_accumulate(rr[i*2+j],pp[i],qq[j],rr[i*2+j]);
            }
        }
    }
    #pragma unroll
    for(uint i=0;i<2;i++) {
        #pragma unroll
        for(uint j=0;j<2;j++) {
            save<256,256,false,true>(cc[i*2+j],p,c,row+i*8,col+j*8,lane,g.x!=g.y);
            save<256,256,false,true>(rr[i*2+j],p,r,row+i*8,col+j*8,lane,g.x!=g.y);
        }
    }
}
kernel void update(device const half* b,device const half* x,device half* y,uint3 g [[threadgroup_position_in_grid]],
 uint sg [[simdgroup_index_in_threadgroup]],uint l [[thread_index_in_simdgroup]]) {
    mm<256,704,false,false>(b,x,y,g,sg,l);
}
kernel void norm(device const float* g, device float* norms,
    uint t [[thread_index_in_threadgroup]], uint b [[threadgroup_position_in_grid]]) {
    threadgroup float sums[256];
    float s = 0.0f;
    for (uint i=t; i<180224; i+=256) { float v=g[b*180224+i]; s+=v*v; }
    sums[t]=s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d=128; d>0; d>>=1) {
        if (t<d) sums[t]+=sums[t+d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(t==0) norms[b]=1.0f/(sqrt(sums[0])+1e-7f);
}
kernel void normalize(device const float* g,device const float* norms,device half* x,
    uint3 block [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
    threadgroup half tile[32*33];
    for(uint j=t;j<1024;j+=256) {
        uint r=block.x*32+j/32,c=block.y*32+j%32;
        tile[(j/32)*33+j%32]=half(g[block.z*180224+r*256+c]*norms[block.z]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=t;j<1024;j+=256) {
        uint r=block.y*32+j/32,c=block.x*32+j%32;
        x[block.z*180224+r*704+c]=tile[(j%32)*33+j/32];
    }
}
""")


def kernel(g):
    """Five Newton-Schulz steps, specialized for (10, 704, 256) float inputs."""
    X = torch.empty((10, 256, 704),device=g.device,dtype=torch.float16)
    norms = torch.empty((10,),device=g.device,dtype=torch.float32)
    _lib.norm(g, norms, threads=2560, group_size=256)
    _lib.normalize(g,norms,X,threads=(5632,8,10),group_size=(256,1,1))
    Y = torch.empty_like(X)
    A = torch.empty((10,256,256),device=g.device,dtype=torch.float16)
    B = torch.empty_like(A)
    C = torch.empty_like(A)
    Q = torch.empty_like(A)
    R = torch.empty_like(A)
    # X_i = Q_i X_0. Track A_i = X_i X_i^T with small matrix products:
    # P_i = 3.4445 I - 4.775 A_i + 2.0315 A_i^2,
    # Q_{i+1} = P_i Q_i, A_{i+1} = (P_i A_i) P_i.
    # These matrices commute in exact arithmetic, so all remain symmetric.
    _lib.gram(X,A,threads=(1024,8,10),group_size=(128,1,1))
    for i in range(5):
        _lib.square(A,B,threads=(1024,8,10),group_size=(128,1,1))
        if i == 0:
            Q,B=B,Q
            P=Q
        else:
            if i < 4:
                _lib.dual(B,A,Q,C,R,threads=(1024,8,10),group_size=(128,1,1))
            else:
                _lib.smallsym(B,Q,R,threads=(1024,8,10),group_size=(128,1,1))
            Q,R=R,Q
            P=B
        if i<4:
            if i == 0:
                _lib.smallsym(P,A,C,threads=(1024,8,10),group_size=(128,1,1))
            _lib.smallsym(C,P,A,threads=(1024,8,10),group_size=(128,1,1))
    _lib.update(Q,X,Y,threads=(1024,22,10),group_size=(128,1,1))
    return Y.mT
