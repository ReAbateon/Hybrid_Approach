int16_t dt_rec(Node_Rec *node, int16_t *sample){
	if(sample[node->feature] <= node->threshold){
		if(node->left != &delta){
			return dt_rec(node->left, sample);
		}else{
			return node->left_class;
		}
	}else{
		if(node->right != &delta){
			return dt_rec(node->right, sample);
		}else{
			return node->right_class;
		}
	}
}