int dt_loop(Node* arr, int16_t* sample){
	Node *node = &arr[0];
	while(node->threshold != DELTA){
		if(sample[node->feature] <= node->threshold){
			node = node->u_left.left;
		}else{
			node = node->u_right.right;
		}
	}
	int class = node->u_left.class_l;
	return class;
}