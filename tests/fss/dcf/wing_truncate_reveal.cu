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

#include <stdio.h>
#include <cmath>
#include <cassert>
#include <cstdint>

#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"

#include "fss/dcf/gpu_truncate.h"

using T = u64;

inline T cpuMsb(T x, int bin){
    return ((x >> (bin - 1)) & T(1));
}

int main(int argc, char *argv[]) {
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    // initCommBufs(true);
    int bin = 64;
    int bout = 64;
    int shift = 16;
    int N = atoi(argv[3]);
    int party = atoi(argv[1]);
    
    auto peer = new GpuPeer(false);
    peer->connect(party, argv[2]);

    // generate the share of x + rin
    auto h_X = new T[N];
    auto d_X = randomGEOnGpuWithGap<T>(N, bin, 2);
    // generate rin
    auto d_mask = randomGEOnGpu<T>(N, bin);
    // generate x = x_0 + x_1 - rin
    auto d_masked_X = (T*) gpuMalloc(N * sizeof(T));
    gpuLinearComb(64, N, d_masked_X, T(1), d_X, T(1), d_mask);
    h_X = (T *)moveToCPU((u8 *)d_X, N * sizeof(T), NULL);    
    int bw = 64;

    u8 *startPtr, *curPtr;
    size_t keyBufSz = 10 * OneGB;
    getKeyBuf(&startPtr, &curPtr, keyBufSz);
    T* h_r = (T*) cpuMalloc(N * sizeof(T));
    dcf::TruncateType t = dcf::TruncateType::RevealedStochasticTruncate;

    // generate TReKey
    printf("Generating key\n");
    auto d_truncateMask = dcf::genGPUTruncateKey(&curPtr, party, t, bin, bout, shift, N, d_mask, &g, h_r);
    printf("Key generated\n");
    assert(curPtr - startPtr < keyBufSz);
    auto h_truncateMask = (T*) moveToCPU((u8*) d_truncateMask, N * sizeof(T), NULL);

    
    curPtr = startPtr;
    std::cout << "Reading key\n";
    auto k = dcf::readGPUTruncateKey<T>(t, &curPtr);
    auto h_TRe = new T[N];
    dcf::gpuTruncate(bin, bout, t, k, shift, peer, party, N, d_masked_X, &g, (Stats*) NULL);
    h_TRe = (T*) moveToCPU((u8*) d_masked_X, N * sizeof(T), NULL);
    // 计算结果是存在d_mask_X的
    destroyGPURandomness();

    for (int i = 0; i < N; i++)
    {
        auto unmasked_TRe = h_TRe[i] - h_truncateMask[i];
        cpuMod(unmasked_TRe, bout);
        auto o = cpuArs(h_X[i], bin, shift);
        cpuMod(o, bout);
        // if (o != unmasked_TRe){
            // if (int(o-unmasked_TRe) > 5){
        printf("%d: h_x = %ld, real_truncate = %ld, stTR_res = %ld, diff = %ld\n", i, h_X[i], o, unmasked_TRe, o-unmasked_TRe);
            // }
        // }
    }
    std::cout << peer->peer->keyBuf->bytesSent << std::endl;
}