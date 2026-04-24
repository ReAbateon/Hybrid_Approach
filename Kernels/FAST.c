int16_t fast(Triangle *arr, int16_t *sample){
	//const mve_pred16_t p3 = vctp16q(3);
	int16x8_t twos = {1, 2, 4, 8, 16, 32, 64, 0};
	int16_t out[8];
	int16_t next_index = 1;

	Triangle *tri = &arr[0];

	while(next_index > 0){
		int16x8_t thresh = vld1q_s16(&tri->thresholds[0]);

		uint16x8_t samp_offset = vld1q_u16(&tri->features[0]);

		int16x8_t samp = vldrhq_gather_shifted_offset_s16(sample, samp_offset);

		mve_pred16_t pred = vcmpleq_s16(samp, thresh);
		int16x8_t dst = vpselq_s16(vdupq_n_s16(1), vdupq_n_s16(0), pred);

		int32_t index = vmladavaq_s16(0, dst, twos);

		next_index = tri->M[index];
		tri = tri + next_index;
	}
	return next_index;
}
