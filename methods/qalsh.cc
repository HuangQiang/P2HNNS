#include "qalsh.h"

namespace p2h {

// -----------------------------------------------------------------------------
QALSH::QALSH(                       // constructor
    int n,                              // cardinality
    int d,                              // dimensionality
    int m)                              // number of hash tables
    : n_pts_(n), dim_(d), m_(m)
{
    // generate hash functions
    a_ = new float[m * d];
    for (int i = 0; i < m * d; ++i) a_[i] = gaussian(0.0f, 1.0f);

    // allocate space for hash tables <tables_>
    tables_ = new Result[m * n];
}

// -----------------------------------------------------------------------------
QALSH::~QALSH()                     // destructor
{
    delete[] a_;      a_      = NULL;
    delete[] tables_; tables_ = NULL;
}

// -----------------------------------------------------------------------------
float QALSH::calc_hash_value(       // calc hash value
    int   d,                            // dimension for calc hash value
    int   tid,                          // hash table id
    const float *data)                  // one data/query object
{
    return calc_inner_product(d, &a_[tid*dim_], data);
}

// -----------------------------------------------------------------------------
float QALSH::calc_hash_value(       // calc hash value
    int   d,                            // dimension for calc hash value
    int   tid,                          // hash table id
    const Result *data)                 // sample data
{
    const float *a = &a_[tid * dim_];

    float val = 0.0f;
    for (int i = 0; i < d; ++i) {
        int   idx   = data[i].id_;
        float coord = data[i].key_;
        val += a[idx] * coord;
    }
    return val;
}

// -----------------------------------------------------------------------------
int QALSH::nns(                     // nearest neighbor search
    int   collision_threshold,          // collision threshold
    int   cand,                         // number of candidates
    const float *query,                 // input query object
    std::vector<int> &cand_list)        // candidates (return)
{
    cand_list.clear();
    cand = std::min(cand, n_pts_);
    
    // simply check all data if #candidates is equal to the cardinality
    if (cand == n_pts_) {
        cand_list.resize(n_pts_);
        for (int i = 0; i < n_pts_; ++i) cand_list[i] = i;
        return n_pts_;
    }

    // init parameters
    int  *freq    = new int[n_pts_]; memset(freq, 0, n_pts_*sizeof(int));
    bool *checked = new bool[n_pts_]; memset(checked, false, n_pts_*sizeof(bool));
    bool *flag    = new bool[m_]; memset(flag, true, m_*sizeof(bool));

    int   *l_pos  = new int[m_];
    int   *r_pos  = new int[m_];
    float *q_val  = new float[m_];

    Result tmp;
    Result *table = NULL;
    for (int i = 0; i < m_; ++i) {
        tmp.key_ = calc_hash_value(dim_, i, query);
        q_val[i] = tmp.key_;

        table = &tables_[i*n_pts_];
        int pos = std::lower_bound(table, table+n_pts_, tmp, cmp) - table;
        if (pos == 0) { l_pos[i] = -1;  r_pos[i] = pos; } 
        else { l_pos[i] = pos; r_pos[i] = pos + 1; }
    }

    // -------------------------------------------------------------------------
    //  c-k-ANN search
    // -------------------------------------------------------------------------
    int   cand_cnt = 0;             // candidate counter
    float kdist    = MAXREAL;
    float w        = 1.0f;          // grid width
    float radius   = 1.0f;          // search radius
    float width    = radius * w / 2.0f; // bucket width

    while (true) {
        // ---------------------------------------------------------------------
        //  step 1: initialize the stop condition for current round
        // ---------------------------------------------------------------------
        int num_flag = 0; memset(flag, true, m_*sizeof(bool));

        // ---------------------------------------------------------------------
        //  step 2: (R,c)-NN search
        // ---------------------------------------------------------------------
        while (num_flag < m_) {
            for (int j = 0; j < m_; ++j) {
                if (!flag[j]) continue;

                table = &tables_[j*n_pts_];
                int   cnt = -1, pos = -1;
                float q_v = q_val[j], ldist = -1.0f, rdist = -1.0f;

                // -------------------------------------------------------------
                //  step 2.1: scan the left part of hash table
                // -------------------------------------------------------------
                cnt = 0; pos = l_pos[j];
                while (cnt < SCAN_SIZE) {
                    ldist = MAXREAL;
                    if (pos >= 0) ldist = fabs(q_v - table[pos].key_);
                    else break;
                    if (ldist > width) break;

                    int id = table[pos].id_;
                    if (++freq[id] >= collision_threshold && !checked[id]) {
                        checked[id] = true;
                        cand_list.push_back(id);
                        if (++cand_cnt >= cand) break;
                    }
                    --pos; ++cnt;
                }
                if (cand_cnt >= cand) break;
                l_pos[j] = pos;

                // -------------------------------------------------------------
                //  step 2.2: scan the right part of hash table
                // -------------------------------------------------------------
                cnt = 0; pos = r_pos[j];
                while (cnt < SCAN_SIZE) {
                    rdist = MAXREAL;
                    if (pos < n_pts_) rdist = fabs(q_v - table[pos].key_);
                    else break;
                    if (rdist > width) break;

                    int id = table[pos].id_;
                    if (++freq[id] >= collision_threshold && !checked[id]) {
                        checked[id] = true;
                        cand_list.push_back(id);
                        if (++cand_cnt >= cand) break;
                    }
                    ++pos; ++cnt;
                }
                if (cand_cnt >= cand) break;
                r_pos[j] = pos;

                // -------------------------------------------------------------
                //  step 2.3: check whether this width is finished scanned
                // -------------------------------------------------------------
                if (ldist > width && rdist > width && flag[j]) {
                    flag[j] = false; 
                    if (++num_flag >= m_) break;
                }
            }
            if (num_flag >= m_ || cand_cnt >= cand) break;
        }
        // ---------------------------------------------------------------------
        //  step 3: stop condition
        // ---------------------------------------------------------------------
        if (cand_cnt >= cand) break;

        // ---------------------------------------------------------------------
        //  step 4: auto-update radius
        // ---------------------------------------------------------------------
        radius = radius * 2.0f;
        width  = radius * w / 2.0f;
    }
    // release space
    delete[] freq;
    delete[] l_pos;
    delete[] r_pos;
    delete[] checked;
    delete[] flag;
    delete[] q_val;

    return cand_cnt;
}

// -----------------------------------------------------------------------------
int QALSH::nns(                     // nearest neighbor search
    int   collision_threshold,          // collision threshold
    int   cand,                         // number of candidates
    int   sample_dim,                   // sample dimension
    const Result *query,                // query object
    std::vector<int> &cand_list)        // candidates (return)
{
    cand_list.clear();
    cand = std::min(cand, n_pts_);
    
    // simply check all data if #candidates is equal to the cardinality
    if (cand == n_pts_) {
        cand_list.resize(n_pts_);
        for (int i = 0; i < n_pts_; ++i) cand_list[i] = i;
        return n_pts_;
    }

    // init parameters
    int  *freq    = new int[n_pts_]; memset(freq, 0, n_pts_*sizeof(int));
    bool *checked = new bool[n_pts_]; memset(checked, false, n_pts_*sizeof(bool));
    bool *flag    = new bool[m_]; memset(flag, true, m_*sizeof(bool));

    int   *l_pos  = new int[m_];
    int   *r_pos  = new int[m_];
    float *q_val  = new float[m_];

    Result tmp;
    Result *table = NULL;
    for (int i = 0; i < m_; ++i) {
        tmp.key_ = calc_hash_value(sample_dim, i, query);
        q_val[i] = tmp.key_;

        table = &tables_[i*n_pts_];
        int pos = std::lower_bound(table, table+n_pts_, tmp, cmp) - table;
        if (pos == 0) { l_pos[i] = -1;  r_pos[i] = pos; } 
        else { l_pos[i] = pos; r_pos[i] = pos + 1; }
    }

    // -------------------------------------------------------------------------
    //  c-k-ANN search
    // -------------------------------------------------------------------------
    int   cand_cnt = 0;             // candidate counter
    float w        = 1.0f;          // grid width
    float radius   = 1.0f;          // search radius
    float width    = radius * w / 2.0f; // bucket width

    while (true) {
        // ---------------------------------------------------------------------
        //  step 1: initialize the stop condition for current round
        // ---------------------------------------------------------------------
        int num_flag = 0; memset(flag, true, m_*sizeof(bool));

        // ---------------------------------------------------------------------
        //  step 2: (R,c)-NN search
        // ---------------------------------------------------------------------
        while (num_flag < m_) {
            for (int j = 0; j < m_; ++j) {
                if (!flag[j]) continue;

                table = &tables_[j*n_pts_];
                int   cnt = -1, pos = -1;
                float q_v = q_val[j], ldist = -1.0f, rdist = -1.0f;

                // -------------------------------------------------------------
                //  step 2.1: scan the left part of hash table
                // -------------------------------------------------------------
                cnt = 0; pos = l_pos[j];
                while (cnt < SCAN_SIZE) {
                    ldist = MAXREAL;
                    if (pos >= 0) ldist = fabs(q_v - table[pos].key_);
                    else break;
                    if (ldist > width) break;

                    int id = table[pos].id_;
                    if (++freq[id] >= collision_threshold && !checked[id]) {
                        checked[id] = true;
                        cand_list.push_back(id);
                        if (++cand_cnt >= cand) break;
                    }
                    --pos; ++cnt;
                }
                if (cand_cnt >= cand) break;
                l_pos[j] = pos;

                // -------------------------------------------------------------
                //  step 2.2: scan the right part of hash table
                // -------------------------------------------------------------
                cnt = 0; pos = r_pos[j];
                while (cnt < SCAN_SIZE) {
                    rdist = MAXREAL;
                    if (pos < n_pts_) rdist = fabs(q_v - table[pos].key_);
                    else break;
                    if (rdist > width) break;

                    int id = table[pos].id_;
                    if (++freq[id] >= collision_threshold && !checked[id]) {
                        checked[id] = true;
                        cand_list.push_back(id);
                        if (++cand_cnt >= cand) break;
                    }
                    ++pos; ++cnt;
                }
                if (cand_cnt >= cand) break;
                r_pos[j] = pos;

                // -------------------------------------------------------------
                //  step 2.3: check whether this width is finished scanned
                // -------------------------------------------------------------
                if (ldist > width && rdist > width && flag[j]) {
                    flag[j] = false; 
                    if (++num_flag >= m_) break;
                }
            }
            if (num_flag >= m_ || cand_cnt >= cand) break;
        }
        // ---------------------------------------------------------------------
        //  step 3: stop condition
        // ---------------------------------------------------------------------
        if (cand_cnt >= cand) break;

        // ---------------------------------------------------------------------
        //  step 4: auto-update radius
        // ---------------------------------------------------------------------
        radius = radius * 2.0f;
        width  = radius * w / 2.0f;
    }
    // release space
    delete[] freq;
    delete[] l_pos;
    delete[] r_pos;
    delete[] checked;
    delete[] flag;
    delete[] q_val;

    return cand_cnt;
}

} // end namespace p2h
