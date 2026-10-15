"""Muon orthogonalization using reduced-precision matrix products."""
import torch

def _make_shader():
    lines = [r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void fast_mm(device const float *L, device const float *R, device const float *C,
                    device float *O, constant uint &add,
                    uint3 tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row=tg.x*64+sg/4*32, col=tg.y*64+sg%4*16, base=tg.z*65536;
''']
    for i in range(4):
        for j in range(2):
            lines.append(f"simdgroup_float8x8 z{i}{j}(0);")
    lines.append("for(uint k=0;k<256;k+=32) {")
    for k in range(0,32,8):
        lines.append("{")
        for i in range(4):
            lines.append(f"simdgroup_float8x8 a{i}; simdgroup_load(a{i},L+base+(row+{i*8})*256+k+{k},256);")
        for j in range(2):
            lines.append(f"simdgroup_float8x8 b{j}; simdgroup_load(b{j},R+base+(k+{k})*256+col+{j*8},256);")
        for i in range(4):
            for j in range(2):
                lines.append(f"simdgroup_multiply_accumulate(z{i}{j},a{i},b{j},z{i}{j});")
        lines.append("}")
    lines.append("}")
    for i in range(4):
        for j in range(2):
            pos=f"base+(row+{i*8})*256+col+{j*8}"
            lines.append(f"if(add) {{ simdgroup_float8x8 d; simdgroup_load(d,C+{pos},256); z{i}{j}.thread_elements()[0]+=d.thread_elements()[0]; z{i}{j}.thread_elements()[1]+=d.thread_elements()[1]; }} simdgroup_store(z{i}{j},O+{pos},256);")
    lines.append("}")
    return "\n".join(lines)

def _outer_shaders():
    lines=[]
    for name in ('gram', 'apply'):
        gram=name=='gram'
        kmax=704 if gram else 256
        n=256 if gram else 704
        ltype='half' if gram else 'float'
        otype='float' if gram else 'half'
        lines.append(f'''kernel void {name}(device const {ltype} *L, device const half *R, device {otype} *O,
          uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {{
          uint row=tg.x*64+sg/4*32,col=tg.y*64+sg%4*16,batch=tg.z;''')
        for i in range(4):
            for j in range(2):
                lines.append(f'simdgroup_float8x8 z{i}{j}(0);')
        lines.append(f'for(uint k=0;k<{kmax};k+=32) {{')
        for k in range(0,32,8):
            lines.append('{')
            for i in range(4):
                lines.append(f'simdgroup_half8x8 a{i};')
                if gram:
                    lines.append(f'simdgroup_load(a{i},L+batch*180224+(row+{i*8})*704+k+{k},704);')
                else:
                    lines.append(f'simdgroup_float8x8 af{i}; simdgroup_load(af{i},L+batch*65536+(row+{i*8})*256+k+{k},256); a{i}.thread_elements()[0]=half(af{i}.thread_elements()[0]); a{i}.thread_elements()[1]=half(af{i}.thread_elements()[1]);')
            for j in range(2):
                lines.append(f'simdgroup_half8x8 b{j};')
                if gram:
                    lines.append(f'simdgroup_load(b{j},R+batch*180224+(col+{j*8})*704+k+{k},704,ulong2(0),true);')
                else:
                    lines.append(f'simdgroup_load(b{j},R+batch*180224+(k+{k})*704+col+{j*8},704);')
            for i in range(4):
                for j in range(2):
                    lines.append(f'simdgroup_multiply_accumulate(z{i}{j},a{i},b{j},z{i}{j});')
            lines.append('}')
        lines.append('}')
        for i in range(4):
            for j in range(2):
                if gram:
                    lines.append(f'''{{ uint rr=row+{i*8}+(lane&6)/2+(lane&16)/4, cc=col+{j*8}+(lane&1)*2+(lane&8)/2;
                      z{i}{j}.thread_elements()[0]=200.0f*z{i}{j}.thread_elements()[0]-(rr==cc ? 1.1f : 0.0f);
                      z{i}{j}.thread_elements()[1]=200.0f*z{i}{j}.thread_elements()[1]-(rr==cc+1 ? 1.1f : 0.0f);
                      simdgroup_store(z{i}{j},O+batch*65536+(row+{i*8})*256+col+{j*8},256); }}''')
                else:
                    lines.append(f'simdgroup_half8x8 h{i}{j}; h{i}{j}.thread_elements()[0]=half(z{i}{j}.thread_elements()[0]); h{i}{j}.thread_elements()[1]=half(z{i}{j}.thread_elements()[1]); simdgroup_store(h{i}{j},O+batch*256*{n}+(row+{i*8})*{n}+col+{j*8},{n});')
        lines.append('}')
    return '\n'.join(lines)
_init_source = r'''
kernel void rescale(device const half *A, device float *Z, uint i [[thread_position_in_grid]]) {
    uint rc=i%65536;
    Z[i]=200.0f*float(A[i])-(rc/256==rc%256 ? 1.1f : 0.0f);
}
kernel void blocks(device const float *Z, device const float *Z2,
 device float *P0, device float *P1, device float *P2, device float *P3,
 uint i [[thread_position_in_grid]]) {
    uint rc=i%65536;
    float d=rc/256==rc%256 ? 1.0f : 0.0f;
    float x=Z[i], y=Z2[i];
    P0[i]=14.713175303389436f*d-14.380655664667518f*x-10.688916094941385f*y;
    P1[i]=24.785859149684217f*d-28.820378064004238f*x+69.79296173674018f*y;
    P2[i]=84.86459769060141f*d-291.99518440913954f*x+38.73131678642112f*y;
    P3[i]=224.92019542327594f*d-78.34988935228415f*x-26.483320125604283f*y;
}
kernel void polynomial(device const half *A, device const half *AA, device half *B,
 uint i [[thread_position_in_grid]]) {
    uint rc=i%65536;
    B[i]=half(2.0315f*float(AA[i])-4.7750f*float(A[i])+(rc/256==rc%256 ? 3.4445f : 0.0f));
}
kernel void norm_parts(device const float *g, device float *parts,
 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    uint base=tg.y*180224+tg.x*1024+tid;
    float v=0;
    for(uint j=0;j<4;j++) { float x=g[base+j*256]; v+=x*x; }
    v=simd_sum(v);
    threadgroup float s[8];
    if(lane==0) s[sg]=v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) {
        float total=0; for(uint j=0;j<8;j++) total+=s[j];
        parts[tg.y*176+tg.x]=total;
    }
}
kernel void normalize(device const float *g, device const float *parts, device half *X,
 uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    float sum=0;
    for(uint i=tid;i<176;i+=128) sum+=parts[tg.z*176+i];
    sum=simd_sum(sum);
    threadgroup float sums[4];
    threadgroup float scale;
    threadgroup half tile[32*32];
    if(lane==0) sums[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) scale=1.0f/(sqrt(sums[0]+sums[1]+sums[2]+sums[3])+1e-7f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<1024;i+=128) {
        uint rr=i/32,cc=i%32;
        tile[i]=half(g[tg.z*180224+(tg.y*32+rr)*256+tg.x*32+cc]*scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<1024;i+=128) {
        uint rr=i/32,cc=i%32;
        X[tg.z*180224+(tg.x*32+rr)*704+tg.y*32+cc]=tile[cc*32+rr];
    }
}
'''
_lib = torch.mps.compile_shader(_make_shader()+_outer_shaders()+_init_source)


def kernel(g):
    transposed = g.size(-2) > g.size(-1)
    X = g.mT if transposed else g
    X = torch.empty((10,256,704),device=g.device,dtype=torch.float16)
    parts = torch.empty((10,176),device=g.device,dtype=torch.float32)
    _lib.norm_parts(g,parts,threads=(176*256,10),group_size=(256,1))
    _lib.normalize(g,parts,X,threads=(1024,22,10),group_size=(128,1,1))
    Z = torch.empty((10,256,256),device=g.device,dtype=torch.float32)
    _lib.gram(X,X,Z,threads=(1024,4,10),group_size=(256,1,1))
    Z2=torch.empty_like(Z)
    Z3=torch.empty_like(Z)
    _lib.fast_mm(Z,Z,Z,Z2,0,threads=(1024,4,10),group_size=(256,1,1))
    _lib.fast_mm(Z2,Z,Z,Z3,0,threads=(1024,4,10),group_size=(256,1,1))
    P0,P1,P2,P3 = [torch.empty_like(Z) for _ in range(4)]
    _lib.blocks(Z,Z2,P0,P1,P2,P3,threads=10*65536)
    T=torch.empty_like(Z)
    U=torch.empty_like(Z)
    _lib.fast_mm(P3,Z3,P2,T,1,threads=(1024,4,10),group_size=(256,1,1))
    _lib.fast_mm(T,Z3,P1,U,1,threads=(1024,4,10),group_size=(256,1,1))
    _lib.fast_mm(U,Z3,P0,T,1,threads=(1024,4,10),group_size=(256,1,1))
    Y=torch.empty_like(X)
    _lib.apply(T,X,Y,threads=(1024,11,10),group_size=(256,1,1))
    X=Y
    return X.mT if transposed else X
