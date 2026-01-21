// Example of inference using the Hybrid Approach on a model that fits entirely in DTCM, resulting in full SIMT execution.

#include "hybrid.h"

for(int r = 0; r < TEST_ROWS; r++){
    cyccnt_before = DWT->CYCCNT;
    inference(testset[r]);
    cyccnt_after = DWT->CYCCNT;
    printf("%d: %lu\r\n", r, cyccnt_after - cyccnt_before);
    printf("%d: ", r);
    for (int i = 0; i < GROUPS * PARALLELISM; i++) {
        printf("%d ", final_results[i]);
    }
    printf("\r\n");
}
