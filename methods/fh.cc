#include "fh.h"

namespace p2h {

// -----------------------------------------------------------------------------
Furthest_Hash::Furthest_Hash(		// constructor
	int   n,	    					// number of data objects
	int   d,							// dimension of data objects
	int   m,							// number of hash tables
	int   s, 							// scale factor of dimension
	float b,							// interval ratio
	const float *data) 				    // input data
	: n_pts_(n), dim_(d), scale_(s), b_(b), data_(data)
{
	sample_dim_ = d * s;
	fh_dim_     = d * (d + 1) / 2 + 1;
	M_          = MINREAL;

	// -------------------------------------------------------------------------
	//  calc centroid, l2-norm, and max l2-norm
	// -------------------------------------------------------------------------
	int    fh_dim_1  = fh_dim_ - 1;
	float  *norm     = new float[n];	// l2-norm
	float  *centroid = new float[fh_dim_];
	int    *ctrd_cnt = new int[fh_dim_1];

	float  *prob     = new float[d];
	bool   *checked  = new bool[fh_dim_];
	int    *sample_d = new int[n];
	Result *sample_data = new Result[n * sample_dim_];

	memset(centroid, 0.0f, fh_dim_  * SIZEFLOAT);
	memset(ctrd_cnt, 0,    fh_dim_1 * SIZEINT);

	for (int i = 0; i < n; ++i) {
		transform_data(&data[i*d], prob, checked, norm[i], sample_d[i], 
			&sample_data[i*sample_dim_], centroid, ctrd_cnt);
		if (M_ < norm[i]) M_ = norm[i];
	}
	for (int i = 0; i < fh_dim_1; ++i) centroid[i] /= ctrd_cnt[i];

	for (int i = 0; i < n; ++i) {
		norm[i] = sqrt(M_ - norm[i]);
		centroid[fh_dim_1] += norm[i];
	}
	centroid[fh_dim_1] /= n;

	// -------------------------------------------------------------------------
	//  determine shift_id after shifting data objects to centroid
	// -------------------------------------------------------------------------
	Result *arr = new Result[n];
	for (int i = 0; i < n; ++i) {
		float dist = calc_transform_dist(sample_d[i], 
			&sample_data[i*sample_dim_], centroid);

		arr[i].id_  = i;
		arr[i].key_ = sqrt(dist + SQR(norm[i] - centroid[fh_dim_1]));
	}
	qsort(arr, n, sizeof(Result), ResultCompDesc);

	shift_id_ = new int[n];
	for (int i = 0; i < n; ++i) shift_id_[i] = arr[i].id_;

	// -------------------------------------------------------------------------
	//  divide datasets into blocks and build hash tables for each block
	// -------------------------------------------------------------------------
	int start = 0;
	while (start < n) {
		// partition block
		float min_radius  = b_ * arr[start].key_;
		int   block_index = start, cnt = 0;
		while (block_index < n && arr[block_index].key_ > min_radius) {
			++block_index;
			if (++cnt >= MAX_BLOCK_NUM) break;
		}

		// add block
		const int *index = (const int*) shift_id_ + start;
		RQALSH *hash = new RQALSH(cnt, fh_dim_, m, index);

		for (int i = 0; i < cnt; ++i) {
			// calc hash value
			int idx = index[i];
			for (int j = 0; j < m; ++j) {
				float val = hash->calc_hash_value(sample_d[idx], j, 
					&sample_data[idx * sample_dim_]);
				val += hash->a_[j*fh_dim_+fh_dim_1] * norm[idx];

				hash->tables_[j*cnt+i].id_  = i;
				hash->tables_[j*cnt+i].key_ = val;
			}
		}
		// sort hash tables in ascending order by hash values
		for (int i = 0; i < m; ++i) {
			qsort(&hash->tables_[i*cnt], cnt, sizeof(Result), ResultComp);
		}
		hash_.push_back(hash);
		start += cnt;
	}
	assert(start == n);

	// -------------------------------------------------------------------------
	//  release space
	// -------------------------------------------------------------------------
	delete[] arr;
	delete[] sample_data;
	delete[] sample_d;
	delete[] checked;
	delete[] prob;
	delete[] ctrd_cnt;
	delete[] centroid;
	delete[] norm;
}

// -----------------------------------------------------------------------------
void Furthest_Hash::transform_data(	// data transformation
	const  float *data,					// input data
	float  *prob,						// probability vector
	bool   *checked,					// is checked?
	float  &norm,						// norm of fh_data (return)
	int    &sample_d,					// sample dimension (return)
	Result *sample_data,				// sample data (return)
	float  *centroid,					// centroid (return)
	int    *ctrd_cnt)					// centroid coordinate conuter (return)
{
	// calc probability vector and the l2-norm-square of data
	float norm2 = 0.0f;
	for (int i = 0; i < dim_; ++i) {
		norm2 += data[i] * data[i];
		prob[i] = norm2;
	}
	for (int i = 0; i < dim_; ++i) prob[i] /= norm2;

	// randomly sample coordinate of data as the coordinate of sample_data
	int   tmp_idx, tmp_idy, idx, idy;
	float tmp_key;

	norm     = 0.0f;
	sample_d = 0;
	memset(checked, false, fh_dim_ * SIZEBOOL);
	
	// first select the largest coordinate
	tmp_idx = dim_-1;
	tmp_key = data[tmp_idx] * data[tmp_idx];

	checked[tmp_idx] = true;
	sample_data[sample_d].id_  = tmp_idx;
	sample_data[sample_d].key_ = tmp_key;
	centroid[tmp_idx] += tmp_key; ++ctrd_cnt[tmp_idx];
	norm += tmp_key * tmp_key;
	++sample_d;

	// consider the combination of the left coordinates
	for (int i = 1; i < sample_dim_; ++i) {
		tmp_idx = sampling(dim_-1, prob);
		tmp_idy = sampling(dim_, prob);
		idx = std::min(tmp_idx, tmp_idy); idy = std::max(tmp_idx, tmp_idy);

		if (idx == idy) {
			tmp_idx = idx;
			if (!checked[tmp_idx]) {
				tmp_key = data[idx] * data[idx];
				
				checked[tmp_idx] = true;
				sample_data[sample_d].id_  = tmp_idx;
				sample_data[sample_d].key_ = tmp_key;
				centroid[tmp_idx] += tmp_key; ++ctrd_cnt[tmp_idx];
				norm += tmp_key * tmp_key; 
				++sample_d; 
			}
		}
		else {
			tmp_idx = dim_ + (idx*dim_-idx*(idx+1)/2) + (idy-idx-1);
			if (!checked[tmp_idx]) {
				tmp_key = data[idx] * data[idy];

				checked[tmp_idx] = true;
				sample_data[sample_d].id_  = tmp_idx;
				sample_data[sample_d].key_ = tmp_key;
				centroid[tmp_idx] += tmp_key; ++ctrd_cnt[tmp_idx];
				norm += tmp_key * tmp_key; 
				++sample_d;
			}
		}
	}
}

// -----------------------------------------------------------------------------
int Furthest_Hash::sampling(		// sampling coordinate based on prob
	int   d,							// dimension
	const float *prob)					// input probability
{
	float end = prob[d-1];
	float rnd = uniform(0.0f, end);
	return std::lower_bound(prob, prob + d, rnd) - prob;
}

// -----------------------------------------------------------------------------
float Furthest_Hash::calc_transform_dist( // calc l2-dist-sqr after transform
	int   sample_d,						// dimension of sample data
	const Result *sample_data,			// sample data
	const float *centroid)				// centroid after data transformation
{
	float dist = 0.0f;
	for (int i = 0; i < sample_d; ++i) {
		int idx = sample_data[i].id_;
		dist += SQR(sample_data[i].key_ - centroid[idx]);
	}
	return dist;
}

// -----------------------------------------------------------------------------
Furthest_Hash::~Furthest_Hash()		// destructor
{
	if (!hash_.empty()) {
		for (auto hash : hash_) { delete hash; hash = NULL; }
		hash_.clear(); hash_.shrink_to_fit();
	}
	delete[] shift_id_; shift_id_ = NULL;
}

// -----------------------------------------------------------------------------
void Furthest_Hash::display()		// display parameters
{
	printf("Parameters of FH:\n");
	printf("    n            = %d\n",   n_pts_);
	printf("    dim          = %d\n",   dim_);
	printf("    scale factor = %d\n",   scale_);
	printf("    sample_dim   = %d\n",   sample_dim_);
	printf("    fh_dim       = %d\n",   fh_dim_);
	printf("    b            = %.2f\n", b_);
	printf("    M            = %f\n",   sqrt(M_));
	printf("    #blocks      = %d\n\n", (int) hash_.size());
}

// -----------------------------------------------------------------------------
int Furthest_Hash::nns(				// point-to-hyperplane NNS
    int   top_k,		    			// top-k value
	int   l,							// separation threshold
	int   cand,							// #candidates
    const float *query,		       		// input query
    MinK_List *list)				    // top-k results (return)
{
	// -------------------------------------------------------------------------
	//  query transformation
	// -------------------------------------------------------------------------
	float  norm_sample_q = 0.0f;
	int    sample_d = -1;
	Result *sample_query = new Result[sample_dim_];
	transform_query(query, norm_sample_q, sample_d, sample_query);
	
	// -------------------------------------------------------------------------
	//  point-to-hyperplane NNS
	// -------------------------------------------------------------------------
	int   verif_cnt = 0;
	int   n_cand    = cand + top_k - 1;
	float fix_val   = norm_sample_q + M_;
	std::vector<int> cand_list;
	
	for (auto hash : hash_) {
		// ---------------------------------------------------------------------
		//  check candidates returned by rqalsh
		// ---------------------------------------------------------------------
		float kfn_dist = -1.0f;
		if (list->isFull()) {
			float kdist = list->max_key();
			kfn_dist = sqrt(fix_val - 2 * kdist * kdist);
		}
		// std::min(block_cand_, n_cand)
		int size = hash->fns(l, n_cand, kfn_dist, sample_d, 
			(const Result*) sample_query, cand_list);

		for (int j = 0; j < size; ++j) {
			int   idx  = cand_list[j];
			float dist = fabs(calc_inner_product(dim_, &data_[idx*dim_], query));
			list->insert(dist, idx + 1);
		}

		// ---------------------------------------------------------------------
		//  update info
		// ---------------------------------------------------------------------
		verif_cnt += size; n_cand -= size; 
		if (n_cand <= 0) break;
	}
	delete[] sample_query;

	return verif_cnt;
}

// -----------------------------------------------------------------------------
void Furthest_Hash::transform_query( // query transformation
	const  float *query,				// input query
	float  &norm_sample_q,				// l2-norm sqr of q after transform (return)
	int    &sample_d,					// dimension of sample query (return)
	Result *sample_query)				// sample query after transform (return)
{
	// calc probability vector
	float norm2 = 0.0f;
	float *prob = new float[dim_];
	for (int i = 0; i < dim_; ++i) {
		norm2 += query[i] * query[i];
		prob[i] = norm2;
	}
	for (int i = 0; i < dim_; ++i) prob[i] /= norm2;

	// randomly sample coordinate of query as the coordinate of sample_query
	int   tmp_idx, tmp_idy, idx, idy;
	float tmp_key;
	bool  *checked = new bool[fh_dim_];
	memset(checked, false, fh_dim_ * SIZEBOOL);

	norm_sample_q = 0.0f;
	sample_d = 0;
	for (int i = 0; i < sample_dim_; ++i) {
		tmp_idx = sampling(dim_, prob);
		tmp_idy = sampling(dim_, prob);
		idx = std::min(tmp_idx, tmp_idy); idy = std::max(tmp_idx, tmp_idy);

		if (idx == idy) {
			tmp_idx = idx;
			if (!checked[tmp_idx]) {
				tmp_key = query[idx] * query[idx];

				checked[tmp_idx] = true;
				sample_query[sample_d].id_  = tmp_idx;
				sample_query[sample_d].key_ = tmp_key;
				norm_sample_q += tmp_key * tmp_key;
				++sample_d;
			}
		}
		else {
			tmp_idx = dim_ + (idx*dim_-idx*(idx+1)/2) + (idy-idx-1);
			if (!checked[tmp_idx]) {
				tmp_key = 2 * query[idx] * query[idy];

				checked[tmp_idx] = true;
				sample_query[sample_d].id_  = tmp_idx;
				sample_query[sample_d].key_ = tmp_key;
				norm_sample_q += tmp_key * tmp_key;
				++sample_d;
			}
		}
	}
	// multiply lambda
	float lambda = sqrt(M_ / norm_sample_q);
	norm_sample_q = norm_sample_q / (lambda * lambda);
	for (int i = 0; i < sample_d; ++i) sample_query[i].key_ *= lambda;

	delete[] prob;
	delete[] checked;
}

} // end namespace p2h
