#include <cassert>
#include <cstdint>

#include "utils/gpu_mem.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_random.h"
#include "utils/rss3_comms.h"
// #include "utils/gpu_data_types.h"

#include "fss/gpu_matmul.h"
#include "nn/orca/fc_layer.h"

using T = replicated_secret_share64;

using namespace dcf;
using namespace dcf::orca;

int main(int argc, char *argv[])
{
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    int bin = 64, bout = 64, M = 100, N = 10, K = 64;
    bool useMomentum = true;
    int epoch = 0;

    int party = atoi(argv[1]);
    int port = atoi(argv[3]);
    int port0 = atoi(argv[5]);
    int port1 = atoi(argv[7]);
    printf("party is %d \n", party);
    auto peer = new Rss3Cluster(party, false, false, argv[2], port, argv[4], port0, argv[6], port1);

    printf("\n connected!");
    return 0;
}
