#include "nh_wo_s.h"

namespace p2h {

// -----------------------------------------------------------------------------
NH_LCCS_wo_S::NH_LCCS_wo_S(			// constructor
	int   n,							// number of input data
	int   d,							// dimension of input data
	int   m,							// #hashers
	float w, 							// bucket width
	const float *data)					// input data
	: n_pts_(n), dim_(d), m_(m), w_(w), data_(data), bucketerp_(m, 1)
{
	int proj_size = m * (d * (d + 1) / 2 + 1);
	projv_.resize(proj_size);
	for (int i = 0; i < proj_size; ++i) projv_[i] = gaussian(0.0f, 1.0f);

	projb_.resize(m);
	for(int i = 0; i < m; ++i) projb_[i] = uniform(0, w_);
	
	norms_.resize(n);
	max_norm_= MINREAL;
	get_norms(data);

	// build hash tables for the hash values of data objects
	data_sigs_.resize({n, m});
	SigType **data_sigs_ptr = data_sigs_.to_ptr();
	for (int i = 0; i < n; ++i) {
		// printf("i=%d\n", i);
		get_sig_data(&data[i*d], norms_[i], data_sigs_ptr[i]);
	}
	bucketerp_.build(data_sigs_);
}

// -------------------------------------------------------------------------
void NH_LCCS_wo_S::get_norms(		// get norm of each trasfromed vectors
	const float *data)					// input data
{
	for (int j = 0; j < n_pts_; ++j) {
		// calc square
		int cnt = 0;
		float norm2 = 0.0f, norm4 = 0.0f;
		for (int i = 0; i < dim_; ++i) {
			float tmp = data[i] * data[i];
			norm2 += tmp; norm4 += tmp * tmp;
		}
		norms_[j] = (norm2 * norm2 + norm4) / 2.0f;

		if (max_norm_ < norms_[j]) max_norm_ = norms_[j];
	}
}

// -------------------------------------------------------------------------
NH_LCCS_wo_S::~NH_LCCS_wo_S()		// destructor
{
}

// -----------------------------------------------------------------------------
void NH_LCCS_wo_S::display()		// display parameters
{
	printf("Parameters of NH_LCCS_wo_S:\n");
	printf("    n      = %d\n",   n_pts_);
	printf("    dim    = %d\n",   dim_);
	printf("    m      = %d\n",   m_);
	printf("    w      = %f\n\n", w_);
}

// -------------------------------------------------------------------------
void NH_LCCS_wo_S::get_sig_data(	// get signature of data
	const float *data, 					// input data
	float norm,							// l2-norm of input data
	SigType *sig) const 				// signature (return)
{
	// the dimension of sig is l_
	for (int mm = 0; mm < m_; ++mm) {
		int pidx = (mm)*((dim_*(dim_-1))/2 + 1);
		float projection = 0.0f;
		for (int i = 0; i < dim_; ++i) {
			projection += projv_[pidx] * data[i] * data[i];
			pidx++;
		}
		// calc differential
		for (int i = 0; i < dim_; ++i) {
			for (int j = i + 1; j < dim_; ++j) {
				projection += projv_[pidx] * data[i] * data[j];
				pidx++;
			}
		}
		//norm 
		projection += projv_[pidx] * sqrt(max_norm_-norm);
		pidx++;
		sig[mm] = SigType((projection + projb_[mm])/w_);
	}
}

// -------------------------------------------------------------------------
void NH_LCCS_wo_S::get_sig_query(	// get signature of query
	const float *query, 				// input query
	SigType *sig) const 				// signature (return)
{
	// the dimension of sig is l_
	double norm_q = 0.;
	for (int i = 0; i < dim_; ++i) {
		for (int j = 0; j < dim_; ++j) {
			norm_q = (query[i] * query[j])*(query[i] * query[j]);
		}
	}
	double lambda = max_norm_/norm_q;
	for (int mm = 0; mm < m_; ++mm) {
		int pidx = (mm)*((dim_*(dim_-1))/2 + 1);
		float projection = 0.0f;
		for (int i = 0; i < dim_; ++i) {
			projection += - projv_[pidx] * query[i] * query[i] * lambda;
			pidx++;
		}
		// calc differential
		for (int i = 0; i < dim_; ++i) {
			for (int j = i + 1; j < dim_; ++j) {
				projection += -2 * projv_[pidx] * query[i] * query[j] * lambda;
				pidx++;
			}
		}
		sig[mm] = SigType((projection + projb_[mm])/w_);
	}
}

// -------------------------------------------------------------------------
int NH_LCCS_wo_S::nns(				// point-to-hyperplane NNS
	int   top_k,						// top-k value
	int   cand,							// #candidates
    const float *query,					// input query
	MinK_List *list)					// top-k results (return)
{
	std::vector<SigType> sigs(m_);
	get_sig_query(query, &sigs[0]);
	int verif_cnt = 0;

	// printf("cand=%d\n", cand);
	int step = (cand+m_-1)/m_;
	bucketerp_.for_candidates(step, sigs, [&](int idx) {
		// verify the true distance of idx
		float dist = fabs(calc_inner_product(dim_, &data_[idx*dim_], query));
		list->insert(dist, idx + 1);
		++verif_cnt;
	});
	return verif_cnt;
}

} // end namespace p2h
