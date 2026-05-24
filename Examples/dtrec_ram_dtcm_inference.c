// Example of inference using DT Rec on a model that fits entirely within DTCM.

#include "dt_rec.h"

for(int r = 0; r < TEST_ROWS; r++){
    cyccnt_before = DWT->CYCCNT;
    inference_ram(testset[r]);
    cyccnt_after = DWT->CYCCNT;
    printf("%d: %lu\r\n", r, cyccnt_after - cyccnt_before);
    printf("%d: ", r);
    for (int i = 0; i < TREES; i++) {
        printf("%d ", out[i]);
    }
    printf("\r\n");
}

for(int r = 0; r < TEST_ROWS; r++){
    cyccnt_before = DWT->CYCCNT;
    inference_dtcm(testset[r]);
    cyccnt_after = DWT->CYCCNT;
    printf("%d: %lu\r\n", r, cyccnt_after - cyccnt_before);
    printf("%d: ", r);
    for (int i = 0; i < TREES; i++) {
        printf("%d ", out[i]);
    }
    printf("\r\n");
}