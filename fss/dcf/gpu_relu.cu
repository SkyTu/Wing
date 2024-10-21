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

#include "gpu_relu.h"
#include "utils/gpu_comms.h"

namespace dcf
{
    template <typename T>
    __global__ void genSelectExtKernel(T* inputMask, T* outputMask, T* rm, u8* rd, T* rmd, T* rmu, T* m, T* ud, T* v, T* w, T* z, int bin, int bout, int N){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < N){
            m[i] = (1ULL << bin)* (gpuMsb(inputMask[i], bin)) + rm[i];
            ud[i] = rd[i] ^ 1;
            assert(ud[i] == 0 || ud[i] == 1);
            v[i] = ud[i] * inputMask[i] + outputMask[i];
            w[i] = ud[i] * (rm[i] + inputMask[i]) + outputMask[i];
            z[i] = 2 * ud[i] * (rm[i] + inputMask[i]);
            rmd[i] = rm[i] * rd[i];
            rmu[i] = rm[i] * ud[i];
        }
    }

    template <typename T>
    T* gpuKeyGenSelectExt(uint8_t** key_as_bytes, int party, int bin, int bout, int N, u8* rd, T* inputMask){
        auto rm = randomGEOnGpu<T>(N, bout);
        auto outputMask = randomGEOnGpu<T>(N, bout);
        T* m = (T*)gpuMalloc(N * sizeof(T));
        T* ud = (T*)gpuMalloc(N * sizeof(T));
        T* v = (T*)gpuMalloc(N * sizeof(T));
        T* w = (T*)gpuMalloc(N * sizeof(T));
        T* z = (T*)gpuMalloc(N * sizeof(T));
        T* rmd = (T*)gpuMalloc(N * sizeof(T));
        T* rmu = (T*)gpuMalloc(N * sizeof(T));
        genSelectExtKernel<<<(N - 1) / 256 + 1, 256>>>(inputMask, outputMask, rm, rd, rmd, rmu, m, ud, v, w, z, bin, bout, N);
        writeShares<T, T>(key_as_bytes, party, N, rm, bout);
        writeShares<T, T>(key_as_bytes, party, N, rmd, bout);
        writeShares<T, T>(key_as_bytes, party, N, rmu, bout);
        writeShares<T, T>(key_as_bytes, party, N, ud, bout);
        writeReconstructed<T>(key_as_bytes, m, N);
        writeShares<T, T>(key_as_bytes, party, N, v, bout);
        writeShares<T, T>(key_as_bytes, party, N, w, bout);
        writeShares<T, T>(key_as_bytes, party, N, z, bout);
        writeShares<T, T>(key_as_bytes, party, N, inputMask, bout);
        gpuFree(rm);
        gpuFree(m);
        gpuFree(ud);
        gpuFree(v);
        gpuFree(w);
        gpuFree(z);
        gpuFree(rmd);
        gpuFree(rmu);
        return outputMask;
    }

    // need to check this
    // drelu mask is used as input mask for the next set of protocols
    // do we need something better than u64?
    template <typename T>
    u8 *keygenDRelu(uint8_t **key_as_bytes, int party, int bin, int N, T *d_rin, AESGlobalContext *gaes)
    {
        // need to write everything in the proper format
        // printf("%d, %d\n", bin, N);
        gpuKeyGenDCF<T>(key_as_bytes, party, bin, 1, N, d_rin, T(1), gaes);
        auto d_dreluMask = randomGEOnGpu<u8>(N, 1);
        writeShares<u8, u8>(key_as_bytes, party, N, d_dreluMask, 1);
        return d_dreluMask;
    }
    // need to check this
    template <typename T>
    std::pair<u8 *, T *> gpuGenTwoRoundReluKey(uint8_t **key_as_bytes, int party, int bin, int bout, int N, T *d_inputMask, AESGlobalContext *gaes)
    {
        writeInt(key_as_bytes, bin);
        writeInt(key_as_bytes, bout);
        writeInt(key_as_bytes, N);
        auto d_dreluMask = keygenDRelu(key_as_bytes, party, bin, N, d_inputMask, gaes);
        auto d_outputMask = gpuKeyGenSelect<T, T, u8>(key_as_bytes, party, N, d_inputMask, d_dreluMask, bout);
        return std::make_pair(d_dreluMask, d_outputMask);
    }

    template <typename T>
    std::pair<u32 *, T *> gpuTwoRoundRelu(SigmaPeer *peer, int party, GPU2RoundReLUKey<T> k, T *d_I, AESGlobalContext *gaes, Stats *s)
    {
        std::vector<u32 *> h_dreluMask = {k.dreluKey.dReluMask};
        auto d_drelu = gpuDcf<T, 2, dReluPrologue, dReluEpilogue<false>>(k.dreluKey.dcfKey, party, d_I, gaes, s, &h_dreluMask);
        peer->reconstructInPlace(d_drelu, 1, k.N, s);
        auto d_relu = gpuSelect<T, T, 0, 0>(peer, party, k.bout, k.selectKey, (u32 *)d_drelu, d_I, s, true);
        return std::make_pair(d_drelu, d_relu);
    }

    template <typename T>
    __global__ void reluExtendMuxKernel(int party, int bin, /*int f,*/ int N, T *x, T* y, T *oneHot, T *outMask, u32 *drelu, u32 *xLTRin)
    {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j < N)
        {
            int posInBlock = threadIdx.x & 0xf;
            u32 d = (((u32 *)drelu)[j / 16] >> (2 * posInBlock)) & 3;
            u32 w = (((u32 *)xLTRin)[j / 16] >> (2 * posInBlock)) & 3;
            u32 i = (2 * d + w) & 3;
            // should i store this table transposed instead?
            // will always access sequential elements so might benefit from locality within a thread
            T rotatedP3 = oneHot[4 * j + ((2 - i) & 3)];
            T rotatedP4 = oneHot[4 * j + ((3 - i) & 3)];
            T xIn = x[j];

            y[j] = xIn * rotatedP3 + (xIn + (1ULL << (bin))) * rotatedP4 + outMask[2 * j + (d & 1)];
            u64 dreluBit = static_cast<u64>(d & 1);
            writePackedOp(xLTRin, dreluBit, 1, N);
        }
    }

    template <typename T>
    T* gpuReluExtendMux(int party, int bin, int N,
                              T *d_I, T *h_oneHot, T *h_outMask, u32 *d_drelu,
                              u32 *d_xLTRin, Stats *s)
    {
        auto d_out = (T*) gpuMalloc(N * sizeof(T));
        auto d_oneHot = (T *)moveToGPU((uint8_t *)h_oneHot, 4 * N * sizeof(T), s);
        auto d_outMask = (T *)moveToGPU((uint8_t *)h_outMask, 2 * N * sizeof(T), s);
        reluExtendMuxKernel<<<(N - 1) / 128 + 1, 128>>>(party, bin, N, d_I, d_out, d_oneHot, d_outMask, d_drelu, d_xLTRin);
        checkCudaErrors(cudaDeviceSynchronize());
        gpuFree(d_oneHot);
        gpuFree(d_outMask);
        return d_out;
    }

    template <typename T>
    __global__ void reluExtendMuxKeyKernel(int bin, int bout, int N, T *d_inputMask, u8 *d_dreluMask, u8 *d_dcfMask, T *d_randomMask, T *d_oneHot, T *d_outMask)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            auto onePos = (-(2 * d_dreluMask[i] + d_dcfMask[i])) & T(3);
            assert(onePos < 4);
            for (int j = 0; j < 4; j++)
            {
                d_oneHot[4 * i + j] = (j == onePos ? T(1) : T(0));
            }
            int outputMask0Idx = d_dreluMask[i] & T(1);
            int outputMask1Idx = 1 - outputMask0Idx;
            d_outMask[2 * i + outputMask0Idx] = d_randomMask[i];
            d_outMask[2 * i + outputMask1Idx] = d_randomMask[i] - d_inputMask[i];
            d_dreluMask[i] &= T(1);
        }
    }

    template <typename T>
    T *genReluExtendMuxKey(uint8_t **key_as_bytes, int party, int bin, int bout, int N, T *d_inputMask, u8 *d_dreluMask, u8 *d_dcfMask)
    {
        auto d_randomMask = randomGEOnGpu<T>(N, bout);
        auto d_oneHot = (T *)gpuMalloc(4 * N * sizeof(T));
        auto d_outMask = (T *)gpuMalloc(2 * N * sizeof(T));
        reluExtendMuxKeyKernel<<<(N - 1) / 256 + 1, 256>>>(bin, bout, N, d_inputMask, d_dreluMask, d_dcfMask, d_randomMask, d_oneHot, d_outMask);
        writeShares<T, T>(key_as_bytes, party, 4 * N, d_oneHot, bout);
        writeShares<T, T>(key_as_bytes, party, 2 * N, d_outMask, bout);
        gpuFree(d_oneHot);
        gpuFree(d_outMask);
        return d_randomMask;
    }

    template <typename T>
    std::pair<u8*, T*> gpuKeyGenReluZeroExt(uint8_t **key_as_bytes, int party, int bin, int bout, int N, T *d_inputMask, AESGlobalContext* g)
    {
        writeInt(key_as_bytes, bin);
        writeInt(key_as_bytes, bout);
        writeInt(key_as_bytes, N);
        auto d_dReluMask = dpf::gpuKeyGenDRelu(key_as_bytes, party, bin, N, d_inputMask, g);
        auto d_outputMask = gpuKeyGenSelectExt(key_as_bytes, party, bin, bout, N, d_dReluMask, d_inputMask);
        return std::make_pair(d_dReluMask, d_outputMask);
    }

    template <typename T>
    std::pair<u8 *, T *> gpuKeygenReluExtend(uint8_t **key_as_bytes, int party, int bin, int bout, int N, T *d_inputMask, AESGlobalContext* g)
    {
        writeInt(key_as_bytes, bin);
        writeInt(key_as_bytes, bout);
        writeInt(key_as_bytes, N);
        gpuKeyGenDCF(key_as_bytes, party, bin, 2, N, d_inputMask, T(1), g);
        auto d_dreluMask = randomGEOnGpu<u8>(N, 2);
        // checkCudaErrors(cudaMemset(d_dreluMask, 0, N));
        auto d_dcfMask = randomGEOnGpu<u8>(N, 2);
        // checkCudaErrors(cudaMemset(d_dcfMask, 0, N));
        writeShares<u8, u8>(key_as_bytes, party, N, d_dreluMask, 2);
        writeShares<u8, u8>(key_as_bytes, party, N, d_dcfMask, 2);
        auto d_randomMask = genReluExtendMuxKey(key_as_bytes, party, bin, bout, N, d_inputMask, d_dreluMask, d_dcfMask);
        // gpuFree(d_inputMask);
        gpuFree(d_dcfMask);
        // gpuFree(d_dreluMask);
        return std::make_pair(d_dreluMask, d_randomMask);
    }

    template <typename T>
    std::pair<u32 *, T *> gpuReluExtend(SigmaPeer *peer, int party, GPUReluExtendKey<T> k, T *d_I, AESGlobalContext *g, Stats *s)
    {
        std::vector<u32 *> h_masks = {k.dReluKey.dReluMask, k.dcfMask};
        auto d_dcf = gpuDcf<T, 2, dReluPrologue, dReluEpilogue<true>>(k.dReluKey.dcfKey, party, d_I, g, s, &h_masks);
        peer->reconstructInPlace(d_dcf, 2, 2 * k.dReluKey.dcfKey.memSzOut * 4, s);
        auto d_drelu = d_dcf;
        auto d_xLTRin = (u32 *)(((u8 *)d_dcf) + k.dReluKey.dcfKey.memSzOut);
        auto d_relu = gpuReluExtendMux(party, k.bin, k.N, d_I, k.oneHot, k.outMask, d_drelu, d_xLTRin, s);
        peer->reconstructInPlace(d_relu, k.bout, k.N, s);
        return std::make_pair(d_drelu, d_relu);
    }
    
    template <typename T>
    __global__ void ReluZeroExtMuxKernel(int party, int bin, int bout, int N, T* d_I, u32* d_dcf, T* d_rm, T* d_rmd, T* d_rmu, T* d_ud, T* d_m, T* d_v, T* d_w, T* d_z, T* d_rin, T* res)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            auto msb_x = gpuMsb(d_I[i], bin);
            int laneId = threadIdx.x & 0x1f;
            auto dhat = ((d_dcf[i / 32] >> laneId) & 1ULL);
            auto that = d_m[i] * (1 ^ msb_x);
            assert(dhat == 0 || dhat == 1);
            if(dhat){
                res[i] = d_ud[i] * (that + d_I[i]) + d_w[i] - d_z[i] + msb_x * d_rmu[i];
            }
            else{
                res[i] = (party == SERVER1) * (that + d_I[i]) - d_ud[i] * (that + d_I[i]) + d_v[i] - d_rin[i] - (1 ^ msb_x) * d_rmd[i];
            }
        }
    }
    // 有必要开d_relu这个空间吗？
    template <typename T>
    T* gpuReluZeroExtMux(int party, int bin, int bout, int N, GPUSelectExtKey<T> k, T* d_I, u32* d_dcf, Stats *s){
        auto d_relu = (T*)gpuMalloc(N * sizeof(T));
        auto d_rm = (T *)moveToGPU((uint8_t *)k.rm, N * sizeof(T), s);
        auto d_rmd = (T *)moveToGPU((uint8_t *)k.rmd, N * sizeof(T), s);
        auto d_rmu = (T *)moveToGPU((uint8_t *)k.rmu, N * sizeof(T), s);
        auto d_ud = (T *)moveToGPU((uint8_t *)k.ud, N * sizeof(T), s);
        auto d_m = (T *)moveToGPU((uint8_t *)k.m, N * sizeof(T), s);
        auto d_v = (T *)moveToGPU((uint8_t *)k.v, N * sizeof(T), s);
        auto d_w = (T *)moveToGPU((uint8_t *)k.w, N * sizeof(T), s);
        auto d_z = (T *)moveToGPU((uint8_t *)k.z, N * sizeof(T), s);
        auto d_rin = (T *)moveToGPU((uint8_t *)k.rin, N * sizeof(T), s);
        ReluZeroExtMuxKernel<<<(N - 1) / 256 + 1, 256>>>(party, bin, bout, N, d_I, d_dcf, d_rm, d_rmd, d_rmu, d_ud, d_m, d_v, d_w, d_z, d_rin, d_relu);
        gpuFree(d_rm);
        gpuFree(d_rmd);
        gpuFree(d_rmu);
        gpuFree(d_ud);
        gpuFree(d_m);
        gpuFree(d_v);
        gpuFree(d_w);
        gpuFree(d_z);
        return d_relu;
    }

    template <typename T>
    std::pair<u32 *, T *> gpuReluZeroExt(SigmaPeer *peer, int party, GPUReluZeroExtKey<T> k, T *d_I, AESGlobalContext *g, Stats *s)
    {
        std::vector<u32 *> h_mask({k.dReluKey.mask});
        auto d_dcf = dpf::gpuDcf<T, 1, dpf::dReluPrologue<0>, dpf::dReluEpilogue<0, false>>(k.dReluKey.dpfKey, party, d_I, g, s, &h_mask);
        peer->reconstructInPlace(d_dcf, 1, k.N, s); 
        auto d_relu = gpuReluZeroExtMux(party, k.bin, k.bout, k.N, k.selectKey, d_I, d_dcf, s);
        peer->reconstructInPlace(d_relu, k.bout, k.N, s);
        // 也许这里d_dcf返回的和我们想要的东西不一样，为什么要返回d_relu而不是直接用d_I呢？可能这个数要存起来？
        return std::make_pair(d_dcf, d_relu);
    }
}