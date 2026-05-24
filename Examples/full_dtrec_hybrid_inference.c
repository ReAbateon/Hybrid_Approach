// Example of inference using a full DT Rec Hybrid Approach, with DT Rec execution in DTCM and DT Rec execution in SRAM.

#include "dtrec.h"

for(int r = 0; r < TEST_ROWS; r++){
    cyccnt_before = DWT->CYCCNT;
    inference(testset[r]);
    cyccnt_after = DWT->CYCCNT;
    printf("%d: %lu\r\n", r, cyccnt_after - cyccnt_before);
    printf("%d: ", r);
    for (int i = 0; i < TREES; i++) {
        printf("%d ", out[i]);
    }
    printf("\r\n");
}