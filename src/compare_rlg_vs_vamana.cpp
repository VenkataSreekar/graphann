// compare_rlg_vs_vamana.cpp
//
// Standalone head-to-head benchmark.
// Builds BOTH a Vamana-style index and the RLG index on the same data,
// then sweeps L_search and reports side-by-side Pareto curves.
//
// This file is self-contained: it includes a minimal Vamana implementation
// so you can compare without modifying the original GraphANN codebase.
//
// Usage: ./compare <data.fbin> <queries.fbin> <gt.ibin>
//                  [R=32] [m=2.0] [L_build=100] [K=10]

#include "radial_layer_index.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================
//  Minimal Vamana implementation for comparison
// =============================================================
namespace vamana_ref {

using Cand = std::pair<float, int>;

static float l2sq(const float* a, const float* b, int d) {
    float s = 0; for (int i=0;i<d;++i){float x=a[i]-b[i];s+=x*x;} return s;
}

struct VamanaIndex {
    int n, dim, R;
    float alpha;
    std::vector<float> data;
    std::vector<std::vector<int>> adj;
    int start;

    std::vector<Cand> greedy(const float* q, int L) const {
        std::vector<bool> vis(n, false);
        std::set<Cand> cands, res;
        float d0 = l2sq(q, data.data()+start*(size_t)dim, dim);
        cands.insert({d0, start}); res.insert({d0, start}); vis[start]=true;
        while (!cands.empty()) {
            auto [d, u] = *cands.begin(); cands.erase(cands.begin());
            if ((int)res.size()>=L && d > std::prev(res.end())->first) break;
            for (int nb : adj[u]) {
                if (vis[nb]) continue; vis[nb]=true;
                float dn = l2sq(q, data.data()+nb*(size_t)dim, dim);
                if ((int)res.size()<L || dn<std::prev(res.end())->first) {
                    res.insert({dn,nb}); cands.insert({dn,nb});
                    if ((int)res.size()>L) res.erase(std::prev(res.end()));
                }
            }
        }
        return {res.begin(), res.end()};
    }

    std::vector<int> robust_prune(int p, std::vector<Cand>& cands) const {
        std::vector<int> sel;
        const float* pd = data.data()+p*(size_t)dim;
        for (auto& [d, c] : cands) {
            if ((int)sel.size() >= R) break;
            bool ok = true;
            const float* cd = data.data()+c*(size_t)dim;
            for (int s : sel) {
                float ds = l2sq(cd, data.data()+s*(size_t)dim, dim);
                if (alpha * ds <= d) { ok=false; break; }
            }
            if (ok) sel.push_back(c);
        }
        return sel;
    }

    void build(const std::vector<float>& d, int n_, int dim_, int R_, float a_, int L_) {
        n=n_; dim=dim_; R=R_; alpha=a_;
        data=d; adj.assign(n,{});
        std::vector<std::mutex> mtx(n);

        // medoid start
        std::vector<float> cent(dim,0);
        for (int i=0;i<n;++i) for (int j=0;j<dim;++j) cent[j]+=d[i*(size_t)dim+j];
        for (auto& x:cent) x/=n;
        float best=1e30f; start=0;
        for (int i=0;i<n;++i){float dd=l2sq(cent.data(),d.data()+i*(size_t)dim,dim);if(dd<best){best=dd;start=i;}}

        std::vector<int> ord(n); std::iota(ord.begin(),ord.end(),0);
        std::mt19937 rng(42); std::shuffle(ord.begin(),ord.end(),rng);

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic,64)
#endif
        for (int idx=0; idx<n; ++idx) {
            int p = ord[idx];
            auto cands = greedy(data.data()+p*(size_t)dim, L_);
            std::vector<Cand> cv(cands.begin(), cands.end());
            cv.erase(std::remove_if(cv.begin(),cv.end(),[p](auto& c){return c.second==p;}),cv.end());
            auto nb = robust_prune(p, cv);
            {std::lock_guard<std::mutex> lk(mtx[p]); adj[p]=nb;}
            for (int q2 : nb) {
                std::lock_guard<std::mutex> lk(mtx[q2]);
                if (std::find(adj[q2].begin(),adj[q2].end(),p)==adj[q2].end()) {
                    adj[q2].push_back(p);
                    if ((int)adj[q2].size()>R) {
                        std::vector<Cand> cv2;
                        const float* qd=data.data()+q2*(size_t)dim;
                        for(int x:adj[q2]) cv2.push_back({l2sq(qd,data.data()+x*(size_t)dim,dim),x});
                        std::sort(cv2.begin(),cv2.end());
                        adj[q2]=robust_prune(q2,cv2);
                    }
                }
            }
        }
    }

    std::vector<int> search(const float* q, int K, int L) const {
        auto c = greedy(q, L);
        std::vector<int> r; for (int i=0;i<std::min(K,(int)c.size());++i) r.push_back(c[i].second);
        return r;
    }
};

} // namespace vamana_ref

// =============================================================
//  File I/O
// =============================================================

static std::vector<float> load_fbin(const std::string& p, int& n, int& d) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: "+p);
    f.read((char*)&n, sizeof(int)); f.read((char*)&d, sizeof(int));
    std::vector<float> v((size_t)n*d);
    f.read((char*)v.data(), v.size()*sizeof(float));
    return v;
}

static std::vector<std::vector<int>> load_ibin(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: "+p);
    int nq, K; f.read((char*)&nq, sizeof(int)); f.read((char*)&K, sizeof(int));
    std::vector<std::vector<int>> gt(nq, std::vector<int>(K));
    for (int i=0;i<nq;++i) f.read((char*)gt[i].data(), K*sizeof(int));
    return gt;
}

static float recall(const std::vector<std::vector<int>>& res,
                    const std::vector<std::vector<int>>& gt, int K) {
    double s=0; int nq=res.size();
    for (int i=0;i<nq;++i) {
        int h=0;
        for (int r:res[i]) for (int j=0;j<K&&j<(int)gt[i].size();++j) if(r==gt[i][j]){h++;break;}
        s+=h;
    }
    return (float)(s/(nq*K));
}

// =============================================================
//  Benchmark one index at multiple L values
// =============================================================

template<typename Index>
void sweep(const Index& idx, const std::vector<float>& queries, int nq,
           const std::vector<std::vector<int>>& gt, int K,
           const std::string& label)
{
    std::vector<int> Ls = {10,20,40,60,100,150,200,300,500};
    std::cout << "\n--- " << label << " ---\n";
    std::cout << std::left
              << std::setw(8)  << "L"
              << std::setw(12) << "Recall@K"
              << std::setw(12) << "QPS"
              << "\n" << std::string(32,'-') << "\n";

    int dim = idx.dim;  // works for both via template
    (void)dim;

    for (int L : Ls) {
        std::vector<std::vector<int>> results(nq);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i=0;i<nq;++i)
            results[i] = idx.search(queries.data()+(size_t)i*idx.dim, K, L);
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double,std::micro>(t1-t0).count();
        float  r  = recall(results, gt, K);
        double qps = 1e6 * nq / us;
        std::cout << std::setw(8)  << L
                  << std::setw(12) << std::fixed << std::setprecision(4) << r
                  << std::setw(12) << std::fixed << std::setprecision(1) << qps << "\n";
        if (r > 0.9999f) break;
    }
}

// Specialization for RLG (dim is a method, not field)
void sweep_rlg(const rlg::RadialLayerIndex& idx,
               const std::vector<float>& queries, int nq,
               const std::vector<std::vector<int>>& gt, int K)
{
    std::vector<int> Ls = {10,20,40,60,100,150,200,300,500};
    std::cout << "\n--- Radial Layer Graph ---\n";
    std::cout << std::left
              << std::setw(8)  << "L"
              << std::setw(12) << "Recall@K"
              << std::setw(12) << "QPS"
              << std::setw(14) << "AvgCands"
              << "\n" << std::string(46,'-') << "\n";

    for (int L : Ls) {
        std::vector<std::vector<int>> results(nq);
        float total_cands = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i=0;i<nq;++i) {
            results[i] = idx.search(queries.data()+(size_t)i*idx.dim(), K, L);
            total_cands += idx.last_avg_candidates;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double us  = std::chrono::duration<double,std::micro>(t1-t0).count();
        float  r   = recall(results, gt, K);
        double qps = 1e6 * nq / us;
        std::cout << std::setw(8)  << L
                  << std::setw(12) << std::fixed << std::setprecision(4) << r
                  << std::setw(12) << std::fixed << std::setprecision(1) << qps
                  << std::setw(14) << std::fixed << std::setprecision(1) << total_cands/nq << "\n";
        if (r > 0.9999f) break;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: compare <data.fbin> <queries.fbin> <gt.ibin> "
                     "[R=32] [m=2.0] [L_build=100] [K=10]\n";
        return 1;
    }

    int n, dim, nq, dq;
    auto data    = load_fbin(argv[1], n, dim);
    auto queries = load_fbin(argv[2], nq, dq);
    auto gt      = load_ibin(argv[3]);

    int   R       = argc>4 ? std::stoi(argv[4]) : 32;
    float m       = argc>5 ? std::stof(argv[5]) : 2.0f;
    int   L_build = argc>6 ? std::stoi(argv[6]) : 100;
    int   K       = argc>7 ? std::stoi(argv[7]) : 10;

    std::cout << "=== Head-to-Head: Vamana vs. Radial Layer Graph ===\n"
              << "n=" << n << " dim=" << dim << " nq=" << nq
              << " R=" << R << " m=" << m << " L_build=" << L_build << "\n";

    // ---- Build Vamana ----
    std::cout << "\n[1/2] Building Vamana...\n";
    vamana_ref::VamanaIndex vamana;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        vamana.build(data, n, dim, R, 1.2f, L_build);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "  Vamana build time: "
                  << std::chrono::duration<double>(t1-t0).count() << " s\n";
    }

    // ---- Build RLG ----
    std::cout << "\n[2/2] Building Radial Layer Graph...\n";
    rlg::RadialLayerIndex rlg_idx;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        rlg_idx.build(data, n, dim, R, m, L_build, 0.5f, true);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "  RLG build time: "
                  << std::chrono::duration<double>(t1-t0).count() << " s\n";
    }

    // ---- Degree stats ----
    {
        float mean, std_, gini;
        rlg_idx.degree_stats(mean, std_, gini);
        std::cout << "\nRLG degree: mean=" << mean
                  << " std=" << std_ << " gini=" << gini << "\n";

        // Vamana degree stats
        float vm=0,vs=0,vg=0;
        std::vector<float> degs(n);
        for(int i=0;i<n;++i) degs[i]=(float)vamana.adj[i].size();
        for(float d:degs) vm+=d; vm/=n;
        for(float d:degs) vs+=(d-vm)*(d-vm); vs=std::sqrt(vs/n);
        std::sort(degs.begin(),degs.end());
        float si=0,sd=0; for(int i=0;i<n;++i){si+=(i+1)*degs[i];sd+=degs[i];}
        vg=(2*si)/(n*sd)-(n+1.f)/n;
        std::cout << "Vamana degree: mean=" << vm
                  << " std=" << vs << " gini=" << vg << "\n";
    }

    // ---- Search sweep ----
    // Vamana
    {
        std::vector<int> Ls = {10,20,40,60,100,150,200,300,500};
        std::cout << "\n--- Vamana ---\n"
                  << std::left << std::setw(8) << "L"
                  << std::setw(12) << "Recall@K"
                  << std::setw(12) << "QPS\n"
                  << std::string(32,'-') << "\n";
        for (int L:Ls) {
            std::vector<std::vector<int>> res(nq);
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i=0;i<nq;++i)
                res[i]=vamana.search(queries.data()+(size_t)i*dim, K, L);
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double,std::micro>(t1-t0).count();
            float r = recall(res,gt,K);
            std::cout << std::setw(8)<<L << std::setw(12)<<std::fixed<<std::setprecision(4)<<r
                      << std::setw(12)<<std::fixed<<std::setprecision(1)<<1e6*nq/us<<"\n";
            if(r>0.9999f) break;
        }
    }

    // RLG
    sweep_rlg(rlg_idx, queries, nq, gt, K);

    std::cout << "\n=== Done. Copy the two tables into your report for the Pareto plot. ===\n";
    return 0;
}
