typedef struct{
    uint16_t class[PARALLELISM];
}classes;

bool all_active(mve_pred16_t mask){
	return (mask == 0xFFFF);
}

classes kernel_simt(int16_t* threshold, uint16_t* features, uint16_t* childL, uint16_t* childR, int16_t* sample){
	uint16x8_t a = vld1q_u16(&childL[0]);
	uint16x8_t b = vld1q_u16(&childR[0]);

	int16x8_t thresh = vld1q_s16(&threshold[0]);
	uint16x8_t samp_offset = vld1q_u16(&features[0]);
	int16x8_t samp = vldrhq_gather_shifted_offset_s16(sample, samp_offset);

	mve_pred16_t mask = vcmpeqq_u16((uint16x8_t)vdupq_n_u16(0), (uint16x8_t)vdupq_n_u16(1));

	while(!all_active(mask)){

		mve_pred16_t pred = vcmpleq_s16(samp, thresh);

		uint16x8_t dst = vpselq_u16(a, b, pred);

		thresh = vldrhq_gather_shifted_offset_s16(threshold, dst);

		mask = vcmpeqq_n_s16(thresh, DELTA);

		samp_offset = vldrhq_gather_shifted_offset_u16(features, dst);
		samp = vldrhq_gather_shifted_offset_s16(sample, samp_offset);
		a = vldrhq_gather_shifted_offset_u16(childL, dst);
		b = vldrhq_gather_shifted_offset_u16(childR, dst);
	}

	classes result;
	vst1q_u16(result.class, b);

	return result;
}