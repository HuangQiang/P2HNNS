#include "mh.h"

namespace p2h {

// -----------------------------------------------------------------------------
MH_Hash::MH_Hash(					// constructor
	int   n,							// number of data objects
	int   d,							// dimension of data objects
	int   M, 							// #proj vecotr used for a single hasher
	int   m,							// #single hasher of the compond hasher
	int   l,							// #hash tables
	const int   *index,					// index of input data
	const float *data)					// input data
	: n_pts_(n), dim_(d), M_(M), m_(m), l_(l), index_(index), buckets_(n, l)
{
	// sample random projection variables
	int size = m * l * M * d;
	projv_.resize(size);
	for (int i = 0; i < size; ++i) {
		projv_[i] = gaussian(0.0f, 1.0f);
	}

	// build hash table for the hash values of data objects
	std::vector<SigType> sigs(l);
	for (int i = 0; i < n; ++i) {
		get_sig_data(&data[i*d], sigs);
		buckets_.insert(i, sigs);
	}
}

// -----------------------------------------------------------------------------
MH_Hash::~MH_Hash()					// destructor
{
}

// -----------------------------------------------------------------------------
void MH_Hash::get_sig_data(			// get signature of data
	const float *data, 					// input data
	std::vector<SigType> &sig) const 	// signature (return)
{
	// the dimension of sig is l_
	for (int ll = 0; ll < l_; ++ll) {
		SigType cur_sig = 0;
		for (int mm = 0; mm < m_; ++mm) {
			float product = 1.0f;
			for (int nproj = 0; nproj < M_; ++nproj) {
				float projection = 0.0f;
				for (int i = 0; i < dim_; ++i) {
					int pidx = (ll*m_+mm)*M_*dim_ + nproj*dim_ + i;
					projection += data[i]*projv_[pidx];
				}
				product *= projection;
			}
			SigType new_sig = (product > 0);
			cur_sig = (cur_sig<<1) | new_sig;
		}
		sig[ll] = cur_sig;
	}
}

// -----------------------------------------------------------------------------
void MH_Hash::get_sig_query(		// get signature of query
	const float *query, 				// input query
	std::vector<SigType> &sig) const 	// signature (return)
{
	// the dimension of sig is l_
	for (int ll = 0; ll < l_; ++ll) {
		SigType cur_sig = 0;
		for (int mm = 0; mm < m_; ++mm) {
			float product = 1.0f;
			for (int nproj = 0; nproj < M_; ++nproj) {
				float projection = 0.0f;
				for (int i = 0; i < dim_; ++i) {
					int pidx = (ll*m_+mm)*M_*dim_ + nproj*dim_ + i;
					projection += query[i] * projv_[pidx];
				}
				product *= projection;
			}
			SigType new_sig = !(product > 0);
			cur_sig = (cur_sig<<1) | new_sig;
		}
		sig[ll] = cur_sig;
	}
}

// -----------------------------------------------------------------------------
int MH_Hash::nns(					// point-to-hyperplane NNS
	int   cand,							// #candidates
	const float *data, 	            	// input data
    const float *query,					// input query
	MinK_List *list)					// top-k results (return)
{
	std::vector<SigType> sigs(l_);
	get_sig_query(query, sigs);
	int verif_cnt = 0;

	buckets_.for_cand(cand, sigs, [&](int idx) {
		// verify the true distance of idx
		int   did  = index_[idx];
		float dist = fabs(calc_inner_product(dim_, &data[did*dim_], query));
		list->insert(dist, did + 1);
		++verif_cnt;
	});
	return verif_cnt;
}

} // end namespace p2h
