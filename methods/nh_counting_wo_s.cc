#include "nh_counting_wo_s.h"

namespace p2h {

// -----------------------------------------------------------------------------
NH_Counting_wo_S::NH_Counting_wo_S(	// constructor
	int   n,	    					// number of data objects
    int   d,							// dimension of data objects
    int   m,							// #hash tables
    const float *data) 				    // input data
	: n_pts_(n), dim_(d), data_(data)
{
	nh_dim_ = d * (d + 1) / 2 + 1;
	M_      = MINREAL;
	lsh_    = new QALSH(n, nh_dim_, m);
	
	// -------------------------------------------------------------------------
	//  build hash tables for qalsh
	// -------------------------------------------------------------------------
	int   nh_dim_1 = nh_dim_ - 1;
	float *norm    = new float[n];
	float *nh_data = new float[nh_dim_1];

	for (int i = 0; i < n; ++i) {
		// data transformation
		transform_data(&data[i*d], norm[i], nh_data);
		if (M_ < norm[i]) M_ = norm[i];

		// calc partial hash value
		for (int j = 0; j < m; ++j) {
			float val = lsh_->calc_hash_value(nh_dim_1, j, nh_data);
			lsh_->tables_[j*n+i].id_  = i;
			lsh_->tables_[j*n+i].key_ = val;
		}
	}
	// calc the final hash value
	for (int i = 0; i < n; ++i) {
		float tmp = sqrt(M_ - norm[i]);
		for (int j = 0; j < m; ++j) {
			lsh_->tables_[j*n+i].key_ += lsh_->a_[j*nh_dim_+nh_dim_1] * tmp;
		}
	}
	// sort hash tables in ascending order by hash values
	for (int i = 0; i < m; ++i) {
		qsort(&lsh_->tables_[i*n], n, sizeof(Result), ResultComp);
	}

	// -------------------------------------------------------------------------
	//  release space
	// -------------------------------------------------------------------------
	delete[] nh_data;
	delete[] norm;
}

// -----------------------------------------------------------------------------
void NH_Counting_wo_S::transform_data( // data transformation
	const float *data,					// input data
	float &norm,						// l2-norm-sqr of nh_data (return)
	float *nh_data)						// nh_data (return)
{
	// calc square
	int cnt = 0;
	float norm2 = 0.0f, norm4 = 0.0f;
	for (int i = 0; i < dim_; ++i) {
		float tmp = data[i] * data[i];
		nh_data[cnt++] = tmp; norm2 += tmp; norm4 += tmp * tmp;
	}
	norm = (norm2 * norm2 + norm4) / 2.0f;

	// calc differential
	for (int i = 0; i < dim_; ++i) {
		for (int j = i + 1; j < dim_; ++j) {
			nh_data[cnt++] = data[i] * data[j];
		}
	}
}

// -----------------------------------------------------------------------------
NH_Counting_wo_S::~NH_Counting_wo_S() // destructor
{
	if (lsh_ != NULL) { delete lsh_; lsh_ = NULL; }
}

// -----------------------------------------------------------------------------
void NH_Counting_wo_S::display()	// display parameters
{
	printf("Parameters of NH_Counting_wo_S:\n");
	printf("    n      = %d\n",   n_pts_);
	printf("    dim    = %d\n",   dim_);
	printf("    nh_dim = %d\n",   nh_dim_);
	printf("    m      = %d\n",   lsh_->m_);
	printf("    M      = %f\n\n", sqrt(M_));
}

// -----------------------------------------------------------------------------
int NH_Counting_wo_S::nns(			// point-to-hyperplane NNS
	int   top_k,						// top-k value
	int   l,							// separation threshold
	int   cand,							// #candidates
	const float *query,					// input query
	MinK_List *list)					// top-k results (return)
{
	// query transformation
	float *nh_query = new float[nh_dim_];
	transform_query(query, nh_query);

	// conduct nearest neighbor search by qalsh
	std::vector<int> cand_list;
	int verif_cnt = lsh_->nns(l, cand + top_k - 1, nh_query, cand_list);

	// calc actual distance for candidates returned by qalsh
	for (int i = 0; i < verif_cnt; ++i) {
		int   idx  = cand_list[i];
		float dist = fabs(calc_inner_product(dim_, &data_[idx*dim_], query));
		list->insert(dist, idx + 1);
	}
	delete[] nh_query;

	return verif_cnt;
}

// -----------------------------------------------------------------------------
void NH_Counting_wo_S::transform_query( // query transformation
	const float *query,					// input query
	float *nh_query)					// nh_query after transform (return)
{
	// calc square
	int cnt = 0;
	float norm_q = 0.0f;
	for (int i = 0; i < dim_; ++i) {
		float tmp = -query[i] * query[i];
		nh_query[cnt++] = tmp; norm_q += tmp * tmp;
	}
	// calc differential
	for (int i = 0; i < dim_; ++i) {
		for (int j = i + 1; j < dim_; ++j) {
			float tmp = -2 * query[i] * query[j];
			nh_query[cnt++] = tmp; norm_q += tmp * tmp;
		}
	}
	nh_query[cnt] = 0.0f;

	// multiply lambda
	float lambda = sqrt(M_ / norm_q);
	for (int i = 0; i < cnt; ++i) nh_query[i] *= lambda;
}

} // end namespace p2h
