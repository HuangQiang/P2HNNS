#include "rqalsh.h"

namespace p2h {

// -----------------------------------------------------------------------------
RQALSH::RQALSH(                     // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // #hash tables
    const int *index)
    : n_(n), dim_(d), m_(m), index_(index) 
{
    // generate hash functions
    a_ = new float[m*d];
    for (int i = 0; i < m*d; ++i) a_[i] = gaussian(0.0f, 1.0f);
    
    // allocate space for tables
    tables_ = new Result[m*n];
}

// -----------------------------------------------------------------------------
RQALSH::~RQALSH()                   // destructor
{
    delete[] a_;
    delete[] tables_;
}

// -----------------------------------------------------------------------------
float RQALSH::calc_hash_value(      // calc hash value
    int   d,                            // dimension for calc hash value
    int   tid,                          // hash table ida
    const float *data)                  // input data
{
    const float *a = &a_[tid*dim_];
    return calc_inner_product2<float>(d, data, a);
}

// -----------------------------------------------------------------------------
float RQALSH::calc_hash_value(      // calc hash value
    int   d,                            // dimension for calc hash value
    int   tid,                          // hash table id
    const Result *data)                 // input data
{
    const float *a = &a_[tid*dim_];

    int   idx = -1;
    float val = 0.0f;
    for (int i = 0; i < d; ++i) {
        idx = data[i].id_;
        val += a[idx] * data[i].key_;
    }
    return val;
}

// -----------------------------------------------------------------------------
float RQALSH::calc_hash_value(      // calc hash value
    int   d,                            // dimension for calc hash value
    int   tid,                          // hash table id
    float last,                         // the last coordinate of input data
    const Result *data)                 // input data
{
    const float *a = &a_[tid*dim_];

    int   idx = -1;
    float val = 0.0f;
    for (int i = 0; i < d; ++i) {
        idx = data[i].id_;
        val += a[idx] * data[i].key_;
    }
    return val + a[dim_-1] * last;
}

// -----------------------------------------------------------------------------
int RQALSH::fns(                    // furthest neighbor search
    int   separation_threshold,         // separation threshold
    int   cand,                         // #candidates
    float R,                            // limited search range
    const float *query,                 // query object
    std::vector<int> &cand_list)        // candidates (return)
{
    cand_list.clear();
    cand = std::min(cand, n_);
    
    // simply check all data if #candidates is equal to the cardinality
    if (cand == n_) {
        cand_list.resize(n_);
        for (int i = 0; i < n_; ++i) {
            if (index_) cand_list[i] = index_[i];
            else cand_list[i] = i;
        }
        return n_;
    }

    // init parameters
    int  *freq    = new int[n_];  memset(freq,    0,     n_*sizeof(int));
    bool *checked = new bool[n_]; memset(checked, false, n_*sizeof(bool));
    bool *b_flag  = new bool[m_]; memset(b_flag,  true,  m_*sizeof(bool));
    bool *r_flag  = new bool[m_]; memset(r_flag,  true,  m_*sizeof(bool));

    int   *l_pos  = new int[m_];
    int   *r_pos  = new int[m_];
    float *q_val  = new float[m_];

    for (int i = 0; i < m_; ++i) {
        q_val[i] = calc_hash_value(dim_, i, query);
        l_pos[i] = 0;  
        r_pos[i] = n_ - 1;
    }

    // -------------------------------------------------------------------------
    //  furthest neighbor search
    // -------------------------------------------------------------------------
    int num_range = 0;              // number of search range flag
    int cand_cnt  = 0;              // candidate counter    
    
    float w       = 1.0f;           // grid width
    float radius  = find_radius(w, l_pos, r_pos, q_val); // search radius
    float width   = radius * w / 2.0f; // bucket width
    float range   = R < CHECK_ERROR ? 0.0f : R*w/2.0f; // search range

    while (true) {
        // ---------------------------------------------------------------------
        //  step 1: initialization
        // ---------------------------------------------------------------------
        int num_bucket = 0; memset(b_flag, true, m_*sizeof(bool));

        // ---------------------------------------------------------------------
        //  step 2: (R,c)-FN search
        // ---------------------------------------------------------------------
        while (num_bucket < m_ && num_range < m_) {
            for (int j = 0; j < m_; ++j) {
                // CANNOT add !r_flag[j] as condition, because the
                // r_flag[j] for large radius will affect small radius
                if (!b_flag[j]) continue;

                int    cnt = -1, lpos = -1, rpos = -1;
                float  q_v = q_val[j], ldist = -1.0f, rdist = -1.0f;
                Result *table = &tables_[j*n_];

                // -------------------------------------------------------------
                //  step 2.1: scan left part of bucket
                // -------------------------------------------------------------
                cnt  = 0;
                lpos = l_pos[j]; rpos = r_pos[j];
                while (cnt < SCAN_SIZE) {
                    ldist = MINREAL;
                    if (lpos < rpos) ldist = fabs(q_v - table[lpos].key_);
                    else break;
                    if (ldist < width || ldist < range) break;

                    int id = table[lpos].id_;
                    if (++freq[id] >= separation_threshold && !checked[id]) {
                        checked[id] = true;
                        if (index_) cand_list.push_back(index_[id]);
                        else cand_list.push_back(id);
                        if (++cand_cnt >= cand) break;
                    }
                    ++lpos; ++cnt;
                }
                if (cand_cnt >= cand) break;
                l_pos[j] = lpos;

                // -------------------------------------------------------------
                //  step 2.2: scan right part of bucket
                // -------------------------------------------------------------
                cnt = 0;
                while (cnt < SCAN_SIZE) {
                    rdist = MINREAL;
                    if (lpos < rpos) rdist = fabs(q_v - table[rpos].key_);
                    else break;
                    if (rdist < width || rdist < range) break;

                    int id = table[rpos].id_;
                    if (++freq[id] >= separation_threshold && !checked[id]) {
                        checked[id] = true;
                        if (index_) cand_list.push_back(index_[id]);
                        else cand_list.push_back(id);
                        if (++cand_cnt >= cand) break;
                    }
                    --rpos; ++cnt;
                }
                if (cand_cnt >= cand) break;
                r_pos[j] = rpos;

                // -------------------------------------------------------------
                //  step 2.3: check whether this bucket is finished scanned
                // -------------------------------------------------------------
                if (lpos >= rpos || (ldist < width && rdist < width)) {
                    if (b_flag[j]) { b_flag[j] = false; ++num_bucket; }
                }
                if (lpos >= rpos || (ldist < range && rdist < range)) {
                    if (b_flag[j]) { b_flag[j] = false; ++num_bucket; }
                    if (r_flag[j]) { r_flag[j] = false; ++num_range;  }
                }
                // use break after checking both b_flag and r_flag
                if (num_bucket >= m_ || num_range >= m_) break;
            }
            if (num_bucket >= m_ || num_range >= m_) break;
            if (cand_cnt >= cand) break;
        }
        // ---------------------------------------------------------------------
        //  step 3: stop condition
        // ---------------------------------------------------------------------
        if (num_range >= m_ || cand_cnt >= cand) break;

        // ---------------------------------------------------------------------
        //  step 4: update radius
        // ---------------------------------------------------------------------
        radius = radius / 2.0f;
        width  = radius * w / 2.0f;
    }
    // release space
    delete[] freq;
    delete[] l_pos;
    delete[] r_pos;
    delete[] checked;
    delete[] b_flag;
    delete[] r_flag;
    delete[] q_val;

    return cand_cnt;
}

// -----------------------------------------------------------------------------
float RQALSH::find_radius(          // find proper radius
    float w,                            // grid width                        
    const int *lpos,                    // left  position of query in hash table
    const int *rpos,                    // right position of query in hash table
    const float *q_v)                   // hash value of query
{
    // find projected distance closest to the query in each hash tables 
    std::vector<float> list;
    for (int i = 0; i < m_; ++i) {
        if (lpos[i] < rpos[i]) {
            list.push_back(fabs(tables_[i*n_+lpos[i]].key_ - q_v[i]));
            list.push_back(fabs(tables_[i*n_+rpos[i]].key_ - q_v[i]));
        }
    }
    // sort the array in ascending order 
    std::sort(list.begin(), list.end());

    // find the median distance and return the new radius
    int   num  = (int) list.size();
    float dist = -1.0f;
    if (num % 2 == 0) dist = (list[num / 2 - 1] + list[num / 2]) / 2.0f;
    else dist = list[num / 2];
    
    int kappa = (int) ceil(log(2.0f * dist / w) / log(2.0f));
    return pow(2.0f, kappa);
}

// -----------------------------------------------------------------------------
int RQALSH::fns(                    // furthest neighbor search
    int   separation_threshold,         // separation threshold
    int   cand,                         // #candidates
    float R,                            // limited search range
    int   sample_dim,                   // sample dimension
    const Result *query,                // query object
    std::vector<int> &cand_list)        // candidates (return)
{
    cand_list.clear();
    cand = std::min(cand, n_);

    // simply check all data if #candidates is equal to the cardinality
    if (cand == n_) {
        cand_list.resize(n_);
        for (int i = 0; i < n_; ++i) {
            if (index_) cand_list[i] = index_[i];
            else cand_list[i] = i;
        }
        return n_;
    }

    // init parameters
    int  *freq    = new int[n_]; memset(freq, 0, n_*sizeof(int));
    bool *checked = new bool[n_]; memset(checked, false, n_*sizeof(bool));
    bool *b_flag  = new bool[m_]; memset(b_flag, true, m_*sizeof(bool));
    bool *r_flag  = new bool[m_]; memset(r_flag, true, m_*sizeof(bool));

    int   *l_pos  = new int[m_];
    int   *r_pos  = new int[m_];
    float *q_val  = new float[m_];
    
    for (int i = 0; i < m_; ++i) {
        q_val[i] = calc_hash_value(sample_dim, i, 0.0f, query);
        l_pos[i] = 0;
        r_pos[i] = n_ - 1;
    }

    // -------------------------------------------------------------------------
    //  furthest neighbor search
    // -------------------------------------------------------------------------
    int num_range = 0;                // number of search range flag
    int cand_cnt  = 0;                // candidate counter    
    
    float w       = 1.0f;            // grid width
    float radius  = find_radius(w, l_pos, r_pos, q_val); // search radius
    float width   = radius * w / 2.0f; // bucket width
    float range   = R < CHECK_ERROR ? 0.0f : R*w/2.0f; // search range

    while (true) {
        // ---------------------------------------------------------------------
        //  step 1: initialization
        // ---------------------------------------------------------------------
        int num_bucket = 0; memset(b_flag, true, m_*sizeof(bool));

        // ---------------------------------------------------------------------
        //  step 2: (R,c)-FN search
        // ---------------------------------------------------------------------
        while (num_bucket < m_ && num_range < m_) {
            for (int j = 0; j < m_; ++j) {
                // CANNOT add !r_flag[j] as condition, because the
                // r_flag[j] for large radius will affect small radius
                if (!b_flag[j]) continue;

                int    cnt = -1, lpos = -1, rpos = -1;
                float  q_v = q_val[j], ldist = -1.0f, rdist = -1.0f;
                Result *table = &tables_[j*n_];

                // -------------------------------------------------------------
                //  step 2.1: scan left part of bucket
                // -------------------------------------------------------------
                cnt  = 0;
                lpos = l_pos[j]; rpos = r_pos[j];
                while (cnt < SCAN_SIZE) {
                    ldist = MINREAL;
                    if (lpos < rpos) ldist = fabs(q_v - table[lpos].key_);
                    else break;
                    if (ldist < width || ldist < range) break;

                    int id = table[lpos].id_;
                    if (++freq[id] >= separation_threshold && !checked[id]) {
                        checked[id] = true;
                        if (index_) cand_list.push_back(index_[id]);
                        else cand_list.push_back(id);
                        if (++cand_cnt >= cand) break;
                    }
                    ++lpos; ++cnt;
                }
                if (cand_cnt >= cand) break;
                l_pos[j] = lpos;

                // -------------------------------------------------------------
                //  step 2.2: scan right part of bucket
                // -------------------------------------------------------------
                cnt = 0;
                while (cnt < SCAN_SIZE) {
                    rdist = MINREAL;
                    if (lpos < rpos) rdist = fabs(q_v - table[rpos].key_);
                    else break;
                    if (rdist < width || rdist < range) break;

                    int id = table[rpos].id_;
                    if (++freq[id] >= separation_threshold && !checked[id]) {
                        checked[id] = true;
                        if (index_) cand_list.push_back(index_[id]);
                        else cand_list.push_back(id);
                        if (++cand_cnt >= cand) break;
                    }
                    --rpos; ++cnt;
                }
                if (cand_cnt >= cand) break;
                r_pos[j] = rpos;

                // -------------------------------------------------------------
                //  step 2.3: check whether this bucket is finished scanned
                // -------------------------------------------------------------
                if (lpos >= rpos || (ldist < width && rdist < width)) {
                    if (b_flag[j]) { b_flag[j] = false; ++num_bucket; }
                }
                if (lpos >= rpos || (ldist < range && rdist < range)) {
                    if (b_flag[j]) { b_flag[j] = false; ++num_bucket; }
                    if (r_flag[j]) { r_flag[j] = false; ++num_range;  }
                }
                // use break after checking both b_flag and r_flag
                if (num_bucket >= m_ || num_range >= m_) break;
            }
            if (num_bucket >= m_ || num_range >= m_) break;
            if (cand_cnt >= cand) break;
        }
        // ---------------------------------------------------------------------
        //  step 3: stop condition
        // ---------------------------------------------------------------------
        if (num_range >= m_ || cand_cnt >= cand) break;

        // ---------------------------------------------------------------------
        //  step 4: update radius
        // ---------------------------------------------------------------------
        radius = radius / 2.0f;
        width  = radius * w / 2.0f;
    }
    // release space
    delete[] freq;
    delete[] l_pos;
    delete[] r_pos;
    delete[] checked;
    delete[] b_flag;
    delete[] r_flag;
    delete[] q_val;

    return cand_cnt;
}

} // end namespace p2h
