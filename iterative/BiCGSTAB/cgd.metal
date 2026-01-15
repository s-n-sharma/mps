//CGD-specific kernels

#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

constant int SIMD_WIDTH = 32;
constant int THREADGROUP_SIZE = 256;

kernel void update_xr(
    device float* x [[buffer(0)]],
    device float* r [[buffer(1)]],
    device const float* p [[buffer(2)]],
    device const float* Ap [[buffer(3)]],     
    device const float* r_dot_r [[buffer(4)]], 
    device const float* p_Ap [[buffer(5)]],  
    device atomic_float* r_norm_new [[buffer(6)]],
    constant uint& size [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_simdgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_position_in_threadgroup]]
) {

    float micro_r_sum = 0.0;

    threadgroup float temp[THREADGROUP_SIZE/SIMD_WIDTH];
    
    if (lid < (THREADGROUP_SIZE / SIMD_WIDTH)) {
        temp[lid] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);


    if (gid < size) {
        float alpha = 0.0;

        if (abs(p_Ap[0]) > 1e-9) {
            alpha = r_dot_r[0] / p_Ap[0];
        }

        x[gid] = x[gid] + alpha * p[gid];
        r[gid] = r[gid] - alpha * Ap[gid];
        micro_r_sum = r[gid] * r[gid];
    }

    float local_r_sum = simd_sum(micro_r_sum);

    if (tid == 0) {
        temp[sid] = local_r_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        float thread_group_sum = 0.0f;

        for (int i = 0; i < THREADGROUP_SIZE/SIMD_WIDTH; i++) {
            thread_group_sum += temp[i];
        }

        if (thread_group_sum != 0.0 && !isnan(thread_group_sum)) {
            atomic_fetch_add_explicit(r_norm_new, thread_group_sum, memory_order_relaxed);
        }
    }        
}

kernel void update_p(
    device float* p [[buffer(0)]],
    device const float* r [[buffer(1)]],
    device const float* r_norm_old [[buffer(2)]],
    device const float* r_norm_new [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) {
        return;
    }
    p[gid] = r[gid] + (r_norm_new[0]/r_norm_old[0]) * p[gid];
}

kernel void zero_out(
    device float* in1 [[buffer(0)]],
    device float* in2 [[buffer(1)]]
) {
    in1[0] = 0.0;
    in2[0] = 0.0;
}

