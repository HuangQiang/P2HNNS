#include "fh_wo_s.h"

namespace p2h {

// -----------------------------------------------------------------------------
FH_wo_S::FH_wo_S(					// constructor
	int   n,	    					// number of data objects
	int   d,							// dimension of data objects
	int   m,							// number of hash tables
	float b,							// interval ratio
	const float *data) 				    // input data
	: n_pts_(n), dim_(d), b_(b), data_(data)
{
	fh_dim_ = d * (d + 1) / 2 + 1;
	M_      = MINREAL;

	// -------------------------------------------------------------------------
	//  calc centroid, l2-norm, and max l2-norm
	// -------------------------------------------------------------------------
	int   fh_dim_1  = fh_dim_ - 1;
	float *fh_data  = new float[fh_dim_1];
	float *norm     = new float[n];	// l2-norm
	float *centroid = new float[fh_dim_];

	memset(centroid, 0.0f, fh_dim_ * SIZEFLOAT);
	for (int i = 0; i < n; ++i) {
		calc_transform_centroid(&data[i*d], norm[i], centroid);
		if (M_ < norm[i]) M_ = norm[i];
	}
	for (int i = 0; i < fh_dim_1; ++i) centroid[i] /= n;

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
		float dist = calc_transform_dist(&data[i*d], centroid);

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
			transform_data(&data[idx*d], fh_data);
			for (int j = 0; j < m; ++j) {
				float val = hash->calc_hash_value(fh_dim_1, j, fh_data);
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
	delete[] fh_data;
	delete[] norm;
	delete[] centroid;
	delete[] arr;
}

// -----------------------------------------------------------------------------
void FH_wo_S::calc_transform_centroid( // data transformation with output centroid
	const float *data,					// input data
	float &norm,						// norm of fh_data (return)
	float *centroid)					// fh data (return)
{
	norm = 0.0f;

	// calc square
	int cnt = 0;
	for (int i = 0; i < dim_; ++i) {
		float tmp = data[i] * data[i];
		centroid[cnt++] += tmp; norm += tmp * tmp; 
	}
	// calc differential
	for (int i = 0; i < dim_; ++i) {
		for (int j = i + 1; j < dim_; ++j) {
			float tmp = data[i] * data[j];
			centroid[cnt++] += tmp; norm += tmp * tmp; 
		}
	}
}

// -----------------------------------------------------------------------------
float FH_wo_S::calc_transform_dist( // calc l2-dist after data transformation 
	const float *data,					// input data
	const float *centroid)				// centroid after data transformation
{
	// calc square
	int   cnt  = 0;
	float dist = 0.0f;
	for (int i = 0; i < dim_; ++i) {
		dist += SQR(data[i] * data[i] - centroid[cnt]);
		++cnt;
	}
	// calc differential
	for (int i = 0; i < dim_; ++i) {
		for (int j = i + 1; j < dim_; ++j) {
			dist += SQR(data[i] * data[j] - centroid[cnt]);
			++cnt;
		}
	}
	return dist;
}

// -----------------------------------------------------------------------------
void FH_wo_S::transform_data(		// data transformation
	const float *data,					// input data
	float *fh_data)						// fh data (return)
{	
	// calc square
	int cnt = 0;
	for (int i = 0; i < dim_; ++i) {
		fh_data[cnt++] = data[i] * data[i];
	}
	// calc differential
	for (int i = 0; i < dim_; ++i) {
		for (int j = i + 1; j < dim_; ++j) {
			fh_data[cnt++] = data[i] * data[j];
		}
	}
}

// -----------------------------------------------------------------------------
FH_wo_S::~FH_wo_S()					// destructor
{
	if (!hash_.empty()) {
		for (int i = 0; i < (int) hash_.size(); ++i) {
			delete hash_[i]; hash_[i] = NULL;
		}
		hash_.clear(); hash_.shrink_to_fit();
	}
	delete[] shift_id_; shift_id_ = NULL;
}

// -----------------------------------------------------------------------------
void FH_wo_S::display()				// display parameters
{
	printf("Parameters of FH_wo_S:\n");
	printf("    n       = %d\n",   n_pts_);
	printf("    dim     = %d\n",   dim_);
	printf("    fh_dim  = %d\n",   fh_dim_);
	printf("    b       = %.2f\n", b_);
	printf("    M       = %f\n",   sqrt(M_));
	printf("    #blocks = %d\n\n", (int) hash_.size());
}

// -----------------------------------------------------------------------------
int FH_wo_S::nns(					// point-to-hyperplane NNS
    int   top_k,		    			// top-k value
	int   l,							// separation threshold
	int   cand,							// #candidates
    const float *query,		       		// input query
    MinK_List *list)				    // top-k results (return)
{
	// -------------------------------------------------------------------------
	//  query transformation
	// -------------------------------------------------------------------------
	float *fh_query = new float[fh_dim_];
	float norm_q    = 0.0f;
	transform_query(query, norm_q, fh_query);
	
	// -------------------------------------------------------------------------
	//  point-to-hyperplane NNS
	// -------------------------------------------------------------------------
	int   n_cand    = cand + top_k - 1;
	int   verif_cnt = 0;
	float fix_val   = norm_q + M_;
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
		int size = hash->fns(l, n_cand, kfn_dist, (const float*) fh_query, 
			cand_list); // std::min(block_cand_, n_cand)

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
	delete[] fh_query;

	return verif_cnt;
}

// -----------------------------------------------------------------------------
void FH_wo_S::transform_query(		// query transformation
	const float *query,					// input query
	float &norm_q,						// l2-norm sqr of q after transform (return)
	float *fh_query)					// fh_query after transform (return)
{
	norm_q = 0.0f;

	// calc square
	int cnt = 0;
	for (int i = 0; i < dim_; ++i) {
		float tmp = query[i] * query[i];
		fh_query[cnt++] = tmp; norm_q += tmp * tmp; 
	}
	// calc differential
	for (int i = 0; i < dim_; ++i) {
		for (int j = i + 1; j < dim_; ++j) {
			float tmp = 2 * query[i] * query[j];
			fh_query[cnt++] = tmp; norm_q += tmp * tmp; 
		}
	}
	fh_query[cnt] = 0.0f;

	// multiply lambda
	float lambda = sqrt(M_ / norm_q);
	norm_q = norm_q / (lambda * lambda);
	for (int i = 0; i < cnt; ++i) fh_query[i] *= lambda;
}

} // end namespace p2h
