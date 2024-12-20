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

#include <cassert>
#include "utils/gpu_mem.h"

#include "gpu_sgd.h"

namespace wing
{

    template <typename T>
    __global__ void leftShiftAndAddKernel(T *A, T *B, T *C, int shift, T alpha, int N)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            assert(shift > 0 || alpha > 0);
            
            if(i == 0) {
                printf("%lu %lu %lu %lu %d\n", A[i], B[i], C[i], alpha, shift);
            } 
            C[i] = (A[i] << shift) + alpha * B[i];
            // gpuMod(C[i], wing::global::bw);
            if(i == 0) {
                printf("%lu %lu %lu %lu %d\n", A[i], B[i], C[i], alpha, shift);
            }            
        }
    }

    // gpuLeftShiftAndAdd(N, d_dW, d_Vw, d_Vw, shift, T(wing::mom_fp));
    template <typename T>
    void gpuLeftShiftAndAdd(int N, T *d_A, T *d_B, T *d_C, int shift, T alpha)
    {
        assert(shift < sizeof(T) * 64);
        leftShiftAndAddKernel<<<(N - 1) / 128 + 1, 128>>>(d_A, d_B, d_C, shift, alpha, N);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // global::scale, Vw: 2 * global::scale, dW: 2 * global::scale
    template <typename T>
    void genGpuSGDWithMomentumKey(u8 **key_as_bytes, int party, int bin, int bout, int N, T *h_W, T *d_W,
                                  T *h_Vw, T *d_dW, int scaleW, int scaleVw, int scaledW, TruncateType t, AESGlobalContext *gaes, int epoch)
    {
        size_t memSizeW = N * sizeof(T);
        auto d_Vw = (T *)moveToGPU((u8 *)h_Vw, memSizeW, NULL);
        int shift = wing::mom_scale + scaleVw - scaledW;
        if (wing::lr_scale[epoch] + scaleVw - scaleW > 0){
            std::cout << "-------shift 1 = " << shift << " mom" << T(wing::mom_fp) << "---------" << std::endl;
        }
        gpuLeftShiftAndAdd(N, d_dW, d_Vw, d_Vw, shift, T(wing::mom_fp));
        bool update_bias = (wing::lr_scale[epoch] + scaleVw - scaleW == 0);
        d_Vw = genGPUTruncateKey(key_as_bytes, party, wing::TruncateType::StochasticTruncate, bin, bout, wing::mom_scale, N, d_Vw, gaes);
        moveIntoCPUMem((u8 *)h_Vw, (u8 *)d_Vw /*d_dW*/, memSizeW, NULL);
        //这里应该变成secret share的形式？
        printf("h_Vw=%ld\n", h_Vw[0]);
        bool dWWasNull = false;
        if (d_W == NULL)
        {
            d_W = (T *)moveToGPU((u8 *)h_W, memSizeW, NULL);
            dWWasNull = true;
        }
        shift = wing::lr_scale[epoch] + scaleVw - scaleW;
        auto d_new_W = (T *)gpuMalloc(memSizeW);
        gpuLeftShiftAndAdd(N, d_W, d_Vw, d_new_W, shift, -T(wing::lr_fp));
        if (wing::lr_scale[epoch] + scaleVw - scaleW > 0){
            std::cout << "-------shift 2 = " << shift << "---------" << std::endl;
        }
        if (shift > 0){
            d_new_W = genGPUTruncateKey(key_as_bytes, party, wing::TruncateType::StochasticTruncate, bin, bout, shift, N, d_new_W, gaes);
        }
        moveIntoCPUMem((u8 *)h_W, (u8 *)d_new_W, memSizeW, NULL);
        gpuFree(d_new_W);
        if (dWWasNull)
            gpuFree(d_W);
        gpuFree(d_Vw);
    }

    template <typename T>
    void readGpuSGDWithMomentumKey(TruncateType t, GPUTruncateKey<T> *truncateKeyVw, GPUTruncateKey<T> *truncateKeyW, u8 **key_as_bytes, int scaleW, int scaleVw, int scaledW, int epoch)
    {
        int shift = wing::lr_scale[epoch] + scaleVw - scaleW;
        *truncateKeyVw = readGPUTruncateKey<T>(TruncateType::StochasticTruncate, key_as_bytes);
        if (shift > 0){
            // *truncateKeyVw = readGPUTruncateKey<T>(TruncateType::StochasticTruncate, key_as_bytes);
            *truncateKeyW = readGPUTruncateKey<T>(TruncateType::StochasticTruncate, key_as_bytes);
        }
        // else{
        //     *truncateKeyVw = readGPUTruncateKey<T>(TruncateType::StochasticTruncate, key_as_bytes);
        // }
    }

    template <typename T>
    void gpuSgdWithMomentum(int bin, int bout, int N, T *h_W, T *d_W,
                            T *h_Vw, T *d_dW, int scaleW, int scaleVw, int scaledW, wing::TruncateType t,
                            wing::GPUTruncateKey<T> truncateKeyVw, GPUTruncateKey<T> truncateKeyW, int party, SigmaPeer *peer, AESGlobalContext *gaes, Stats *s, int epoch)
    {
        size_t memSizeW = N * sizeof(T);
        auto d_Vw = (T *)moveToGPU((u8 *)h_Vw, memSizeW, s);
        std::cout << "In gpuSgdWithMomentum h_Vw=" << h_Vw[0] << std::endl;
        int shift = wing::mom_scale + scaleVw - scaledW;
        bool update_bias = (wing::lr_scale[epoch] + scaleVw - scaleW == 0);
        
        if (update_bias){
            gpuLeftShiftAndAdd(N, d_dW, d_Vw, d_Vw, shift, T(wing::mom_fp));
            wing::gpuTruncate(bin, bout, wing::TruncateType::StochasticTruncate, truncateKeyVw, wing::mom_scale, peer, party, N, d_Vw, gaes, s);
        }
        else{
            gpuLeftShiftAndAdd(N, d_dW, d_Vw, d_Vw, shift, T(wing::mom_fp));
            wing::gpuTruncate(bin, bout, wing::TruncateType::StochasticTruncate, truncateKeyVw, wing::mom_scale, peer, party, N, d_Vw, gaes, s, false);
        }
        moveIntoCPUMem((u8 *)h_Vw, (u8 *)d_Vw /*d_dW*/, memSizeW, s);
        std::cout << "h_Vw=" << h_Vw[0] << std::endl;

        bool dWWasNull = false;
        if (d_W == NULL)
        {
            d_W = (T *)moveToGPU((u8 *)h_W, memSizeW, s);  
            dWWasNull = true;
        }
        shift = wing::lr_scale[epoch] + scaleVw - scaleW;
        // this is wrong it needs to be -lr
        if (wing::lr_scale[epoch] + scaleVw - scaleW > 0){
            std::cout << "-------shift 2 = " << shift << "---------" << std::endl;
        }
        if (update_bias){
            gpuLeftShiftAndAdd(N, d_W, d_Vw, d_W, shift, -T(wing::lr_fp));
        }
        else{
            auto d_new_W = (T *)gpuMalloc(memSizeW);
            gpuLinearComb(wing::global::bw, N, d_new_W, T(party), d_W);
            gpuLeftShiftAndAdd(N, d_new_W, d_Vw, d_W, shift, -T(wing::lr_fp));
            wing::gpuTruncate(bin, bout, wing::TruncateType::StochasticTruncate, truncateKeyW, shift, peer, party, N, d_W, gaes, s);
            gpuFree(d_new_W);
        }
        moveIntoCPUMem((u8 *)h_W, (u8 *)d_W, memSizeW, s);
        printf("h_W=%ld\n", h_W[0]);
        if (dWWasNull)
            gpuFree(d_W);
        gpuFree(d_Vw);
    }

    template <typename T>
    void checkSgdWithMomentum(int bin, int bout, int N,
                              T *h_W, T *h_Vw, T *h_dW,
                              T *h_masked_W, T *h_masked_Vw,
                              T *h_mask_W, T *h_mask_Vw,
                              int scaleW, int scaleVw, int scaledW, int epoch)
    {
        int shiftdW = scaleVw + wing::mom_scale - scaledW;
        int shiftW = wing::lr_scale[epoch] + scaleVw - scaleW;
        std::cout << "bin=" << bin << ", bout=" << bout << ", mom_scale=" << wing::mom_scale << ", shiftW=" << shiftW << std::endl;
        for (int i = 0; i < N; i++)
        {
            auto vw = h_masked_Vw[i] - h_mask_Vw[i];
            auto vw_ct = cpuArs((h_dW[i] << shiftdW) + T(wing::mom_fp) * h_Vw[i], bin, wing::mom_scale);
            // if(i < 10) printf("%lu %lu\n", u64(vw), u64(vw_ct));
            // assert(vw - vw_ct <= 1);
            auto w_ct = cpuArs((h_W[i] << shiftW) - T(wing::lr_fp) * vw_ct, bin, shiftW);
            // this is the new masked f
            auto w = h_masked_W[i] - h_mask_W[i];
            // need to test this when the starting vf is non-zero
            auto diff = abs(static_cast<int64_t>(u64(w) - u64(w_ct)));
            if (i < 10)
                printf("%lu %lu h_mask_Vw %lu %ld\n", u64(w), u64(w_ct), u64(h_mask_Vw[i]), diff);
            // the two is important
            // assert(/*abs(static_cast<int64_t>(w - w_ct))*/ diff <= 2);
        }
    }

    template <typename T>
    T *gpuMultiplyByConstant(T *d_A, T x, int N)
    {
        auto d_B = (T *)gpuMalloc(N * sizeof(T));
        gpuLinearComb(sizeof(T) * 8, N, d_B, x, d_A);
        return d_B;
    }

    template <typename T>
    void genGpuSGDKey(u8 **key_as_bytes, int party, int bin, int bout, int N, T *h_W, T *d_W,
                      T *d_dW, int scaleW, int scaledW, TruncateType t, AESGlobalContext *gaes, int epoch)
    {
        size_t memSizeW = N * sizeof(T);
        auto d_delta = gpuMultiplyByConstant(d_dW, -T(wing::lr_fp), N);
        int rightShift = scaledW + wing::lr_scale[epoch] - scaleW;
        bool dWWasNull = false;
        if (rightShift > 0)
        {
            assert(rightShift == wing::global::scale + wing::lr_scale[epoch]);
            d_delta = genGPUTruncateKey(key_as_bytes, party, t, bin, bout, rightShift, N, d_delta, gaes);
            gpuLinearComb(bin, N, d_W, T(1), d_W, T(1), d_delta);
        }
        else
        {
            int leftShift = scaleW - wing::lr_scale[epoch] - scaledW;
            assert(leftShift == wing::global::scale - wing::lr_scale[epoch]);
            assert(d_W == NULL);
            d_W = (T *)moveToGPU((u8 *)h_W, memSizeW, NULL);
            dWWasNull = true;
            gpuLeftShiftAndAdd(N, d_delta, d_W, d_W, leftShift, T(1));
        }
        gpuFree(d_delta);
        moveIntoCPUMem((u8 *)h_W, (u8 *)d_W, memSizeW, NULL);
        if (dWWasNull)
            gpuFree(d_W);
    }

    template <typename T>
    void readGpuSGDKey(TruncateType t, int scaleW, int scaledW, GPUTruncateKey<T> *truncateKeyW, u8 **key_as_bytes, int epoch)
    {
        int rightShift = scaledW + wing::lr_scale[epoch] - scaleW;
        if (rightShift > 0)
        {
            *truncateKeyW = readGPUTruncateKey<T>(t, key_as_bytes);
        }
    }

    template <typename T>
    void gpuSgd(int bin, int bout, int N, T *h_W, T *d_W,
                T *d_dW, int scaleW, int scaledW, TruncateType t,
                GPUTruncateKey<T> truncateKeyW, int party, SigmaPeer *peer, AESGlobalContext *gaes, Stats *s, int epoch)
    {
        size_t memSizeW = N * sizeof(T);
        // the d_dW mask got moved to the left by shift
        auto d_delta = gpuMultiplyByConstant(d_dW, -T(wing::lr_fp), N);
        int rightShift = wing::lr_scale[epoch] + scaledW - scaleW;
        bool dWWasNull = false;
        if (rightShift > 0)
        {
            assert(rightShift == wing::global::scale + wing::lr_scale[epoch]);
            wing::gpuTruncate(bin, bout, t, truncateKeyW, rightShift, peer, party, N, d_delta, gaes, s);
            gpuLinearComb(bin, N, d_W, T(1), d_W, T(1), d_delta);
        }
        else
        {
            int leftShift = scaleW - wing::lr_scale[epoch] - scaledW;
            assert(leftShift == wing::global::scale - wing::lr_scale[epoch]);
            assert(d_W == NULL);
            d_W = (T *)moveToGPU((u8 *)h_W, memSizeW, NULL);
            dWWasNull = true;
            gpuLeftShiftAndAdd(N, d_delta, d_W, d_W, leftShift, T(1));
            // peer->reconstructInPlace(d_W, bout, N, s);
        }
        gpuFree(d_delta);
        moveIntoCPUMem((u8 *)h_W, (u8 *)d_W, memSizeW, s);
        if (dWWasNull)
            gpuFree(d_W);
    }

    template <typename T>
    void checkSgd(int bin, int bout, int N,
                  T *h_W, T *h_dW, T *h_masked_W,
                  T *h_mask_W, int scaleW, int scaledW, int epoch)
    {
        int rightShift = wing::lr_scale[epoch] + scaledW - scaleW;
        if (rightShift > 0)
        {
            assert(rightShift == wing::global::scale + wing::lr_scale[epoch]);
            for (int i = 0; i < N; i++)
            {
                auto w_ct = h_W[i] - cpuArs(T(wing::lr_fp) * h_dW[i], bin, rightShift);
                // this is the new masked f
                auto w = h_masked_W[i] - h_mask_W[i];
                // need to test this when the starting vf is non-zero
                auto diff = abs(static_cast<int32_t>(w - w_ct));
                if (i < 10)
                    printf("%lu %lu %d\n", u64(w), u64(w_ct), diff);
                assert(diff <= 10);
            }
        }
        else
        {
            int leftShift = scaleW - wing::lr_scale[epoch] - scaledW;
            assert(leftShift == wing::global::scale - wing::lr_scale[epoch]);
            for (int i = 0; i < N; i++)
            {
                auto w_ct = h_W[i] - T(wing::lr_fp) * h_dW[i] * (T(1) << leftShift);
                // this is the new masked f
                auto w = h_masked_W[i] - h_mask_W[i];
                // need to test this when the starting vf is non-zero
                auto diff = abs(static_cast<int32_t>(w - w_ct));
                if (i < 10)
                    printf("%lu %lu %ld\n", w, w_ct, diff);
                // assert(diff == 0);
            }
        }
    }

    template <typename T>
    void genOptimizerKey(u8 **key_as_bytes, int party, int bin, int bout, int N, T *h_W, T *d_W,
                         T *h_Vw, T *d_dW, int scaleW, int scaleVw, int scaledW, TruncateType t, bool useMomentum, AESGlobalContext *gaes, int epoch)
    {
        if (useMomentum)
        {
            genGpuSGDWithMomentumKey(key_as_bytes, party, bin, bout, N, h_W, d_W, h_Vw, d_dW, scaleW, scaleVw, scaledW, t, gaes, epoch);
        }
        else
        {
            genGpuSGDKey(key_as_bytes, party, bin, bout, N, h_W, d_W, d_dW, scaleW, scaledW, t, gaes, epoch);
        }
    }

    template <typename T>
    void readOptimizerKey(TruncateType t, GPUTruncateKey<T> *truncateKeyVw, GPUTruncateKey<T> *truncateKeyW, u8 **key_as_bytes, int scaleW, int scaleVw, int scaledW, bool useMomentum, int epoch)
    {
        if (useMomentum)
        {
            readGpuSGDWithMomentumKey(t, truncateKeyVw, truncateKeyW, key_as_bytes, scaleW, scaleVw, scaledW, epoch);
        }
        else
        {
            readGpuSGDKey(t, scaleW, scaledW, truncateKeyW, key_as_bytes, epoch);
        }
    }

    template <typename T>
    void optimize(int bin, int bout, int N, T *h_W, T *d_W,
                  T *h_Vw, T *d_dW, int scaleW, int scaleVw, int scaledW, TruncateType t,
                  GPUTruncateKey<T> truncateKeyVw, GPUTruncateKey<T> truncateKeyW, int party, SigmaPeer *peer, bool useMomentum, AESGlobalContext *gaes, Stats *s, int epoch)
    {
        if (useMomentum)
        {
            gpuSgdWithMomentum(bin, bout, N, h_W, d_W, h_Vw, d_dW, scaleW, scaleVw, scaledW, t, truncateKeyVw, truncateKeyW, party, peer, gaes, s, epoch);
        }
        else
        {
            gpuSgd(bin, bout, N, h_W, d_W, d_dW, scaleW, scaledW, t, truncateKeyW, party, peer, gaes, s, epoch);
        }
    }

    template <typename T>
    void checkOptimizer(int bin, int bout, int N,
                        T *h_W, T *h_Vw, T *h_dW,
                        T *h_masked_W, T *h_masked_Vw,
                        T *h_mask_W, T *h_mask_Vw,
                        int scaleW, int scaleVw, int scaledW, bool useMomentum, int epoch)
    {
        if (useMomentum)
        {
            checkSgdWithMomentum(bin, bout, N, h_W, h_Vw, h_dW, h_masked_W, h_masked_Vw, h_mask_W, h_mask_Vw, scaleW, scaleVw, scaledW, epoch);
        }
        else
        {
            checkSgd(bin, bout, N, h_W, h_dW, h_masked_W, h_mask_W, scaleW, scaledW, epoch);
        }
    }

}