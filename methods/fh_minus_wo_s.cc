#include "fh_minus_wo_s.h"

namespace p2h {

// -----------------------------------------------------------------------------
FH_Minus_wo_S::FH_Minus_wo_S(		// constructor
	int   n,	    					// number of data objects
	int   d,							// dimension of data objects
	int   m,							// #hash tables
	const float *data) 				    // input data
	: n_pts_(n), dim_(d), data_(data)
{
	fh_dim_ = d * (d + 1) / 2 + 1;
	M_      = MINREAL;
	lsh_    = new RQALSH(n, fh_dim_, m, NULL);
	
	// -------------------------------------------------------------------------
	//  build hash tables for rqalsh
	// -------------------------------------------------------------------------
	int   fh_dim_1 = fh_dim_ - 1;
	float *fh_data = new float[fh_dim_1];
	float *norm    = new float[n];	

	for (int i = 0; i < n; ++i) {
		// data transformation
		transform_data(&data[i*d], norm[i], fh_data);
		if (M_ < norm[i]) M_ = norm[i];

		// calc partial hash value
		for (int j = 0; j < m; ++j) {
			float val = lsh_->calc_hash_value(fh_dim_1, j, fh_data);
			lsh_->tables_[j*n+i].id_  = i;
			lsh_->tables_[j*n+i].key_ = val;
		}
	}
	// calc the final hash value
	for (int i = 0; i < n; ++i) {
		float tmp = sqrt(M_ - norm[i]);
		for (int j = 0; j < m; ++j) {
			lsh_->tables_[j*n+i].key_ += lsh_->a_[j*fh_dim_+fh_dim_1] * tmp;
		}
	}
	// sort hash tables in ascending order by hash values
	for (int i = 0; i < m; ++i) {
		qsort(&lsh_->tables_[i*n], n, sizeof(Result), ResultComp);
	}

	// -------------------------------------------------------------------------
	//  release space
	// -------------------------------------------------------------------------
	delete[] fh_data;
	delete[] norm;
}

// -----------------------------------------------------------------------------
void FH_Minus_wo_S::transform_data(	// data transformation
	const float *data,					// input data
	float &norm,						// norm of fh_data (return)
	float *fh_data)						// fh data (return)
{
	// calc square
	int cnt = 0;
	float norm2 = 0.0f, norm4 = 0.0f;
	for (int i = 0; i < dim_; ++i) {
		float tmp = data[i] * data[i];
		fh_data[cnt++] = tmp; norm2 += tmp; norm4 += tmp * tmp;
	}
	norm = (norm2 * norm2 + norm4) / 2.0f;

	// calc differential
	for (int i = 0; i < dim_; ++i) {
		for (int j = i + 1; j < dim_; ++j) {
			fh_data[cnt++] = data[i] * data[j];
		}
	}
}

// -----------------------------------------------------------------------------
FH_Minus_wo_S::~FH_Minus_wo_S()		// destructor
{
	if (lsh_ != NULL) { delete lsh_; lsh_ = NULL; }
}

// -----------------------------------------------------------------------------
void FH_Minus_wo_S::display()		// display parameters
{
	printf("Parameters of FH_Hash:\n");
	printf("    n      = %d\n",   n_pts_);
	printf("    dim    = %d\n",   dim_);
	printf("    fh_dim = %d\n",   fh_dim_);
	printf("    m      = %d\n",   lsh_->m_);
	printf("    M      = %f\n\n", sqrt(M_));
}

// -----------------------------------------------------------------------------
int FH_Minus_wo_S::nns(				// point-to-hyperplane NNS
	int   top_k,						// top-k value
	int   l,							// separation threshold
	int   cand,							// #candidates
	const float *query,					// input query
	MinK_List *list)					// top-k results (return)
{
	// query transformation
	float *fh_query = new float[fh_dim_];
	transform_query(query, fh_query);

	// conduct furthest neighbor search by rqalsh
	std::vector<int> cand_list;
	int verif_cnt = lsh_->fns(l, cand + top_k - 1, MINREAL, 
		(const float *) fh_query, cand_list);

	// calc actual distance for candidates returned by qalsh
	for (int i = 0; i < verif_cnt; ++i) {
		int   idx  = cand_list[i];
		float dist = fabs(calc_inner_product(dim_, &data_[idx*dim_], query));
		list->insert(dist, idx + 1);
	}
	delete[] fh_query;

	return verif_cnt;
}

// -----------------------------------------------------------------------------
void FH_Minus_wo_S::transform_query( // query transformation
	const float *query,					// input query
	float *fh_query)					// fh_query after transform (return)
{
	// calc square
	int cnt = 0;
	float norm_q = 0.0f;
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
	for (int i = 0; i < cnt; ++i) fh_query[i] *= lambda;
}

} // end namespace p2h
