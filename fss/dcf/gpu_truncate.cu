// Author: Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "gpu_truncate.h"
#include "utils/misc_utils.h"
#include "utils/gpu_file_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"

#include "gpu_dcf_templates.h"
#include "fss/gpu_local_truncate.h"
#include <cassert>

namespace dcf
{

    template <typename T>
    __global__ void signExtendKeyKernel(int bin, int bout, int N, T *inMask, u8 *dcfMask, T *t, T *p, T *outMask)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            t[i] = outMask[i] - inMask[i] - (T(1) << (bin - 1));
            gpuMod<T>(t[i], bout);
            assert(dcfMask[i] == 0 || dcfMask[i] == 1);
            int idx0 = dcfMask[i];
            int idx1 = 1 - idx0;
            p[2 * i + idx0] = 0;
            p[2 * i + idx1] = (T(1) << bin);
        }
    }
    
    template <typename T>
    __global__ void keygenStTRKernel(int party, int bin, int bout, int shift, int N, T *inputMask, T *rHat, u8 *lsbMask, T *lsbCorr, T *outMask)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            auto temp = inputMask[i];
            gpuMod(temp, shift);
            auto corr = 1 - 1 * (rHat[i] < temp) - (inputMask[i] >> shift) + outMask[i];
            gpuMod(corr, bout);
            auto corrM1 = corr - 1;
            gpuMod(corrM1, bout);
            lsbCorr[2 * i + lsbMask[i]] = corr;
            lsbCorr[2 * i + (lsbMask[i] ^ 1)] = corrM1;
        }
    }
    
    template <typename T>
    __global__ void keygenTReKernel(int party, int shift, int N, T *inputMask){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            inputMask[i] = (inputMask[i] >> shift);
        }
    }

    // bin = n-f, bout = n, shift = f
    template <typename T>
    __global__ void keygenZeroExtKernel(int party, int bin, int bout, int N, T *inputMask, T *u, T *m, T *outMask)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {   
            u[i] = outMask[i] - inputMask[i];
            gpuMod(u[i], bout);
            m[i] = gpuMsb(inputMask[i], bin) * (1ULL << bin);
            gpuMod(m[i], bout);
        }
    }

    // TReKey现在是本地截断了
    template <typename T>
    T* genGPUTReKey(uint8_t **key_as_bytes, int party, int bin, int bout, int shift, int N, T *d_inputMask, AESGlobalContext *gaes, T *h_r = NULL)
    {
        writeInt(key_as_bytes, bin);
        writeInt(key_as_bytes, bout);
        writeInt(key_as_bytes, shift);
        writeInt(key_as_bytes, N);
        keygenTReKernel<<<(N - 1) / 128 + 1, 128>>>(party, shift, N, d_inputMask);
        return d_inputMask;
    }

    template <typename T>
    T *genGPUZeroExtKey(uint8_t **key_as_bytes, int party, int bin, int bout, int N, T *d_inputMask, AESGlobalContext *gaes)
    {
        writeInt(key_as_bytes, bin);
        writeInt(key_as_bytes, bout);
        writeInt(key_as_bytes, N);
        auto d_outMask = randomGEOnGpu<T>(N, bout);
        auto u = (T*)gpuMalloc(N * sizeof(T));
        auto m = (T*)gpuMalloc(N * sizeof(T));
        keygenZeroExtKernel<<<(N - 1) / 128 + 1, 128>>>(party, bin, bout, N, d_inputMask, u, m, d_outMask);
        writeShares<T, T>(key_as_bytes, party, N, u, bout);
        writeShares<T, T>(key_as_bytes, party, N, m, bout);
        gpuFree(u);
        gpuFree(m);
        return d_outMask;
    }
    
    template <typename T>
    T *genGPUStTRKey(uint8_t **key_as_bytes, int party, int bin, int bout, int shift, int N, T *d_inputMask, AESGlobalContext *gaes, T *h_r = NULL)
    {
        auto d_trMask = genGPUTReKey(key_as_bytes, party, bin, bin - shift, shift, N, d_inputMask, gaes, h_r);
        auto d_outputMask = genGPUZeroExtKey(key_as_bytes, party, bin - shift, bout, N, d_trMask, gaes);
        gpuFree(d_trMask);
        return d_outputMask;
    }


    template <typename T>
    T *genGPUTruncateKey(uint8_t **key_as_bytes, int party, TruncateType t, int bin, int bout, int shift, int N, T *d_inMask, AESGlobalContext *gaes, T *h_r = NULL)
    {
        T *d_outMask;
        switch (t)
        {
        case TruncateType::LocalARS:
            gpuLocalTr<T, T, ars>(party, bin, shift, N, d_inMask, true);
            d_outMask = d_inMask;
            break;
        case TruncateType::StochasticTR:
            bout = bin - shift;
            d_outMask = genGPUTReKey(key_as_bytes, party, bin, bout, shift, N, d_inMask, gaes, h_r);
            break;
        case TruncateType::StochasticTruncate:
            d_outMask = genGPUStTRKey(key_as_bytes, party, bin, bout, shift, N, d_inMask, gaes);
            break;
        default:
            d_outMask = d_inMask;
            assert(t == TruncateType::None);
        }
        return d_outMask;
    }

    template <typename T>
    __global__ void selectForTruncateKernel(T *x, u32 *maskedDcfBit, T *outMask, T *p, int N, int party)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            // can remove the cast to u32* for maskedDcfBit
            // 其实他们是把1比特的数值压缩成u32*进行传输，然后再根据当前执行的threadid来进行位移获得相对应的数
            int dcfBit = (((u32 *)maskedDcfBit)[i / 32] >> (threadIdx.x & 0x1f)) & 1;
            x[i] = (party == SERVER1) * x[i] + outMask[i] + p[2 * i + dcfBit];
        }
    }

    // no memory leak
    template <typename T>
    void gpuSelectForTruncate(int party, int N, T *d_I, u32 *d_maskedDcfBit, T *h_outMask, T *h_p, Stats *s)
    {
        size_t memSz = N * sizeof(T);
        auto d_outMask = (T *)moveToGPU((u8 *)h_outMask, memSz, s);
        auto d_p = (T *)moveToGPU((u8 *)h_p, 2 * memSz, s);
        selectForTruncateKernel<T><<<(N - 1) / 128 + 1, 128>>>(d_I, d_maskedDcfBit, d_outMask, d_p, N, party);
        checkCudaErrors(cudaDeviceSynchronize());
        gpuFree(d_outMask);
        gpuFree(d_p);
    }


    template <typename T>
    __global__ void TReKernel(int party, int bin, int bout, int shift, int N, T *x, bool gap)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {   
            x[i] = (x[i] >> shift);
        }
    }

    template <typename T>
    __global__ void PrintKernel(int idx, T *x)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < idx)
        {   
            printf("%lu\n", x[i]);
        }
    }


    template <typename T>
    __global__ void zeroExtendKernel(T *x, T *m, T *u, int bin, int bout, int N, int party)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {   
            x[i] = x[i] + (1ULL << (bin - 2));
            gpuMod(x[i], bin);
            auto msb_xhat = gpuMsb(x[i], bin);
            x[i] = x[i] - (1ULL << (bin - 2));
            gpuMod(x[i], bout);
            x[i] = party + (party == SERVER1) * x[i] + u[i] + m[i] * (!msb_xhat);
        }
    }

    template <typename T>
    void gpuZeroExtend(int party, int N, int bin, int bout, T *d_I, T *h_m, T *h_u, Stats *s)
    {
        size_t memSz = N * sizeof(T);
        auto d_m = (T *)moveToGPU((u8 *)h_m, memSz, s);
        auto d_u = (T *)moveToGPU((u8 *)h_u, memSz, s);
        zeroExtendKernel<T><<<(N - 1) / 128 + 1, 128>>>(d_I, d_m, d_u, bin, bout, N, party);
        checkCudaErrors(cudaDeviceSynchronize());
        gpuFree(d_m);
        gpuFree(d_u);
    }

    template <typename T>
    void gpuZeroExt(GPUZeroExtKey<T> k, int party, SigmaPeer *peer, T *d_I, AESGlobalContext *g, Stats *s)
    {
        gpuZeroExtend(party, k.N, k.bin, k.bout, d_I, k.m, k.u, s);
        peer->reconstructInPlace(d_I, k.bout, k.N, s);
    }


    template <typename T>
    void gpuTRe(GPUTReKey<T> k, int party, SigmaPeer *peer, T *d_I, AESGlobalContext *g, Stats *s, bool gap = true)
    {   
        TReKernel<<<(k.N - 1) / 128 + 1, 128>>>(party, k.bin, k.bout, k.shift, k.N, d_I, gap);
        peer->reconstructInPlace(d_I, k.bout, k.N, s);
    }

    template <typename T>
    void gpuStTR(GPUTruncateKey<T> k, int party, SigmaPeer *peer, T *d_I, AESGlobalContext *g, Stats *s)
    {
        gpuTRe(k.TReKey, party, peer, d_I, g, s);
        gpuZeroExt(k.ZeroExtKey, party, peer, d_I, g, s);
    }
    

    template <typename T>
    void gpuTruncate(int bin, int bout, TruncateType t, GPUTruncateKey<T> k, int shift, SigmaPeer *peer, int party, int N, T *d_I, AESGlobalContext *gaes, Stats *s)
    {
        switch (t)
        {
        case TruncateType::StochasticTR:
            bout = bin - shift;
            gpuTRe(k.TReKey, party, peer, d_I, gaes, s);
            break;
        case TruncateType::LocalARS:
            gpuLocalTr<T, T, ars>(party, bin, shift, N, d_I, true);
            break;
        case TruncateType::StochasticTruncate:
            gpuStTR(k, party, peer, d_I, gaes, s);
            break;
        default:
            assert(t == TruncateType::None);
        }
        return;
    }
    
    // check via tolerance bounds
    template <typename T>
    void checkTrStWithTol(int bin, int bout, int shift, int N, T *h_masked_A, T *h_mask_A, T *h_A_ct)
    {
        for (int i = 0; i < N; i++)
        {
            auto temp = h_A_ct[i] + T(1ULL << (bin - 1));
            cpuMod(temp, bin);
            auto truncated_A = temp >> shift;
            auto truncated_A_plus1 = truncated_A + 1;
            cpuMod(truncated_A_plus1, bin - shift);
            truncated_A -= T(1ULL << (bin - shift - 1));
            cpuMod(truncated_A, bout);
            truncated_A_plus1 -= T(1ULL << (bin - shift - 1));
            cpuMod(truncated_A_plus1, bout);
            auto output = h_masked_A[i] - h_mask_A[i];
            cpuMod(output, bout);
            if (i < 10)
                printf("%lu %lu %lu\n", h_A_ct[i], u64(output), u64(truncated_A));
            // if (output != truncated_A && output != truncated_A_plus1)
            //     printf("%lu %lu %lu %lu\n", h_A_ct[i], u64(output), u64(truncated_A), u64(truncated_A_plus1));
            // assert(output == truncated_A || output == truncated_A_plus1);
        }
    }
}