int dt_arr(uint16_t* feature, int16_t* threshold, uint16_t* child, int16_t* sample){
    int node = 0;
    uint16_t feat = feature[0];
    while(threshold[node] != DELTA){
        int cmp = sample[feat] <= threshold[node];
        node = cmp*(child[2*node]) + !cmp * (child[(2*node) + 1]);
        feat = feature[node];
    }
    int class = child[2*node];
    return class;
}