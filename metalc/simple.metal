[[kernel]] void
simple_function(
		device   float *x [[buffer(0)]],
		constant float *a [[buffer(1)]],
		         uint   i [[thread_position_in_grid]]
	) {
	x[i] += 1.0f;
	x[i] *= *a;
}
