// Example of inference using the Hybrid Approach, with SIMT execution in DTCM and DT-Rec execution in SRAM.

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
