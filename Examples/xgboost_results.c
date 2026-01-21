// Example of XGBoost Classification De-Quantization

inference();

#if XGBBINARY == 1 // if it is a binary problem
  		  sum = 0;
		  for (int i = 0; i < GROUPS * PARALLELISM; i++){
			  sum += dequantize_uint16(final_results[i], MIN, MAX);
		  }
  		  float p = sigmoidf(sum);
  		  if(p > 0.5){
  			  printf("%d: 1\r\n", r);
  		  }else{
  			  printf("%d: 0\r\n", r);
  		  }
#else // if it is a multi class problem
  		float logit[CLASSES] = {0};
  		float probs[CLASSES] = {0};
  		//printf("%d: ", r);
  		for(int j=0; j < CLASSES; j++){
  			for (int i = 0; i < GROUPS * PARALLELISM; i++){
				logit[j] += dequantize_uint16(final_results[j][i], MIN[j], MAX[j]);
			}
  			 //printf("%f ", logit[j]);
  		}
  		//printf("\r\n");
  		softmax(logit, probs, CLASSES);
//  		printf("%d: ", r);
//  		for (int i = 0; i < CLASSES; i++){
//  			printf("%f ", probs[i]);
//  		}
//  		printf("\r\n");
  		int predicted_class = 0;
  		float max_p = probs[0];
  		for (int k = 1; k < CLASSES; ++k) {
  		    if (probs[k] > max_p) {
  		        max_p = probs[k];
  		        predicted_class = k;
  		    }
  		}
  		printf("%d: %d\r\n", r, predicted_class);
#endif