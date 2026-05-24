// Example of inference using DT Rec

#include "dtrec.h"

for(int r = 0; r < TEST_ROWS; r++){
    cyccnt_before = DWT->CYCCNT;
    inference(testset[r]);
    cyccnt_after = DWT->CYCCNT;
    printf("%d: %lu\r\n", r, cyccnt_after - cyccnt_before);
    printf("%d: ", r);
    for (int i = 0; i < TREES; i++) {
        printf("%d ", classes[i]);
    }
    printf("\r\n");
}
