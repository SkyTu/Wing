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

#include "../../../utils/gpu_data_types.h"
#include "../../../utils/gpu_file_utils.h"
#include "../../../utils/misc_utils.h"
#include "../../../utils/gpu_mem.h"
#include "../../../utils/gpu_random.h"
#include "../../../utils/gpu_comms.h"

#include "../../../fss/dcf/gpu_relu.h"
#include "../../../fss/dcf/gpu_truncate.h"

#include <cassert>
#include <sytorch/tensor.h>

using T = u64;

int main(int argc, char *argv[])
{
    // initCommBufs(true);
    initGPUMemPool();
    AESGlobalContext g;
    initAESContext(&g);
    int bin = 64;
    int bout = 64;
    int shift = 16;
    int N = atoi(argv[3]); //8;
    int party = atoi(argv[1]);

    auto peer = new GpuPeer(true);
    peer->connect(party, argv[2]);

    u8 *startPtr, *curPtr;
    getKeyBuf(&startPtr, &curPtr, 10 * OneGB);
    initGPURandomness();
    auto d_mask_X = randomGEOnGpu<T>(N, bin);
    // checkCudaErrors(cudaMemset(d_mask_X, 0, N * sizeof(T)));
    auto h_mask_X = (T *)moveToCPU((u8 *)d_mask_X, N * sizeof(T), NULL);
    T *h_X;
    auto d_masked_X = getMaskedInputOnGpu(N, bin, d_mask_X, &h_X);
    T* h_r = (T*) cpuMalloc(N * sizeof(T));
    auto d_truncateMask = dcf::genGPUTReKey(&curPtr, party, bin, bin-shift, shift, N, d_mask_X, &g, h_r);
    auto d_reluExtMask = dcf::gpuKeyGenRFSS3ReluZeroExt(&curPtr, party, bin-shift, bout, N, d_truncateMask, &g);
    gpuFree(d_truncateMask);
    printf("Done with keygen\n");
    auto d_dReluMask = d_reluExtMask.first;
    auto d_outputMask = d_reluExtMask.second;
    
    // T* h_dReluMask = (T*)cpuMalloc(N * sizeof(T));
    // h_dReluMask = (T *)moveToCPU((u8 *)d_dReluMask, N * sizeof(T), NULL);
    // printf("finish copying\n");
    T* h_reluOutMask = (T*)cpuMalloc(N * sizeof(T));
    h_reluOutMask = (T *)moveToCPU((u8 *)d_outputMask, N * sizeof(T), NULL);
    printf("finish copying\n");
    // gpuFree(d_dReluMask);
    gpuFree(d_outputMask);
    curPtr = startPtr;
    auto k_TRe = dcf::readGPUTReKey<T>(&curPtr);
    auto k1 = dcf::readRFSS3ReluExtKey<T>(&curPtr);
    T *d_relu;
    for (int i = 0; i < 1; i++)
    {
        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        dcf::gpuTRe(k_TRe, party, peer, d_masked_X, &g, (Stats*) NULL, false);
        auto temp = dcf::gpuRFSS3ReluExtend(peer, party, k1, d_masked_X, &g, (Stats *)NULL);
        auto d_drelu = temp.first;
        gpuFree(d_drelu);
        d_relu = temp.second;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        printf("Time taken=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    }
    auto h_relu = (T *)moveToCPU((u8 *)d_relu, N * sizeof(T), (Stats *)NULL);
    gpuFree(d_relu);
    destroyGPURandomness();
    for (int i = 0; i < N; i++)
    {
        auto unmasked_O = (h_relu[i] - h_reluOutMask[i]);
        cpuMod(unmasked_O, bout);
        auto o = (h_X[i]>>shift) * (1 - (h_X[i] >> (bin - 1)));
        if (i < 10)
            printf("%d: %ld, %ld, %ld\n", i, h_X[i], o, unmasked_O);
        assert(o == unmasked_O || o + 1 == unmasked_O);
    }

    return 0;
}