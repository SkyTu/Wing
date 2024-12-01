// 
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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <omp.h>
#include <unistd.h>

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/helper_cuda.h"
#include "utils/gpu_random.h"

#include "cnn_wing.h"

#include <sytorch/backend/llama_base.h>
#include <sytorch/softmax.h>

u64 *gpuGenSoftmaxKey(int batchSz, int numClasses, u64 *d_mask_I, bool secfloat, LlamaBase<u64> *llama)
{
    Tensor4D<u64> inpMask(batchSz, numClasses, 1, 1);
    Tensor4D<u64> softmaxOpMask(batchSz, numClasses, 1, 1);
    size_t memSz = batchSz * numClasses * sizeof(u64);
    moveIntoCPUMem((u8 *)inpMask.data, (u8 *)d_mask_I, memSz, NULL);
    gpuFree(d_mask_I);
    if (secfloat)
    {
        softmax_secfloat(inpMask, softmaxOpMask, wing::global::scale, 1);
    }
    else
    {
        pirhana_softmax(inpMask, softmaxOpMask, wing::global::scale);
    }
    d_mask_I = (u64 *)moveToGPU((u8 *)softmaxOpMask.data, memSz, NULL);
    return d_mask_I;
}

void genModelKey(wing::GPUModel<u64> *m, u8 **bufPtr, int party, AESGlobalContext *g, bool secfloat, LlamaBase<u64> *llama, int epoch)
{
    auto d_mask_I = randomGEOnGpu<u64>(m->inpSz, wing::global::bw);
    auto h_mask_I = (u64*) moveToCPU((u8*)d_mask_I, m->inpSz * sizeof(u64), NULL);
    printf("Generate Model Key\n");
    for (int i = 0; i < 10; i++){
        printf("h_mask_I[%d] = %lu\n", i, h_mask_I[i]);
    }
    u64 *d_mask_O = NULL;
    for (int i = 0; i < m->layers.size(); i++)
    {   
        d_mask_O = m->layers[i]->genForwardKey(bufPtr, party, d_mask_I, g);
        assert(d_mask_O != d_mask_I);
        gpuFree(d_mask_I);
        d_mask_I = d_mask_O;
    }
    d_mask_I = gpuGenSoftmaxKey(m->batchSz, m->classes, d_mask_I, secfloat, llama);
    for (int i = m->layers.size() - 1; i >= 0; i--)
    {
        d_mask_I = m->layers[i]->genBackwardKey(bufPtr, party, d_mask_I, g, epoch);
    }
}

void writeKeySz(std::string dir, std::string modelName, u64 keySz)
{
    makeDir(dir);
    std::ofstream keySzFile(dir + modelName + ".txt");
    keySzFile << keySz;
    keySzFile.close();
}

void dealerE2E(std::string modelName, int party, int epochs, int blocks, int blockSz, int batchSz, int H, int W, int C, bool secfloat, bool momentum, std::string keyDir, int sleepInt, std::string weightsMask = "", bool fake_offline = true)
{
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    initGPUMemPool();
    sytorch_init();
    // assert(epochs < 6);

    auto expName = modelName + "-" + std::to_string(epochs) + "e-" + std::to_string(blocks) + "b";
    auto trainingDir = "output/P" + std::to_string(party) + "/training/";
    auto lossDir = trainingDir + "loss/" + expName + "/";
    auto keySzDir = trainingDir + "keysize/";
    auto weightsDir = lossDir + "weights/";

    // assumes output/P0/training exists
    makeDir(trainingDir + "loss/");
    makeDir(lossDir);
    makeDir(weightsDir);
    makeDir(keySzDir);

    char one = 1;
    char two = 2;

    std::cout << "before getGPUModel" << std::endl;

    // load the model
    wing::GPUModel<u64> *m = getGPUModel<u64>(modelName, Tensor<u64>(nullptr, {(u64)batchSz, (u64)H, (u64)W, (u64)C}));
    std::cout << "after getGPUModel" << std::endl;
    m->setTrain(momentum);
    m->initWeights(weightsMask, false);

    char *zeros;
    size_t padding, bufSize = 8 * OneGB;
    u8 *startPtr, *curPtr, *tmpPtr1, *tmpPtr2;
    getAlignedBuf(&startPtr, bufSize);

    // initialize llama
    LlamaConfig::party = DEALER;
    auto llama = new LlamaBase<u64>();
    tmpPtr1 = (u8 *)malloc(OneGB);
    bool isServer = party + 2 == SERVER;
    llama->initDealer((char **)(isServer ? &curPtr : &tmpPtr2), (char **)(isServer ? &tmpPtr2 : &curPtr));
    
    std::string keyFile = keyDir + modelName + "_training_key" + std::to_string(party);
    int fd = openForWriting(keyFile + "_" + to_string(0) + "_" + to_string(0) + "_" + std::to_string(0) + ".dat");
    for (int l = 0; l < epochs; l++)
    {
        for (int k = 0; k < blocks; k++)
        {
            printf("Iteration=%u\n", l * blocks * blockSz + k * blockSz);
            for (int j = 0; j < blockSz; j++)
            {
                curPtr = startPtr;
                tmpPtr2 = tmpPtr1;
                genModelKey(m, &curPtr, party, &g, secfloat, (LlamaBase<u64> *)llama, l);
                if (l == 0 && k == 0 && j == 0)
                {
                    size_t keySz = curPtr - startPtr;
                    padding = 4096 - (keySz % 4096);
                    keySz += padding;
                    zeros = new char[padding];
                    memset(zeros, 0, padding);
                    writeKeySz(keySzDir, modelName, keySz);
                }
                memcpy(curPtr, zeros, padding);
                curPtr += padding;
                writeKeyBuf(fd, curPtr - startPtr, startPtr);
            }
            m->dumpWeights(weightsDir + "weights_mask_" + std::to_string(party) + "_" + to_string(l) + "_" + to_string(k) + "_" + std::to_string(blockSz-1) + ".dat");
        }
    }
    close(fd);
    delete[] zeros;
    destroyGPURandomness();
}


int main(int argc, char *argv[])
{
    int party = atoi(argv[1]);
    auto keyDir = std::string(argv[2]);

    omp_set_num_threads(32);
    int epochs = 1;
    int blocks = 46;
    int blockSz = 10; // 600
    int batchSz = 128;
    dealerE2E("CNN2", party, epochs, blocks, blockSz, batchSz, 28, 28, 1, false, false, keyDir, 300, "", true);
    // dealerE2E("P-SecureML", party, epochs, blocks, blockSz, batchSz, 28, 28, 1, false, false, keyDir, 300, "", true);
    return 0;
}
