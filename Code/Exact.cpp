#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <functional>

struct Graph {
    std::vector<std::vector<int>> adj;
    std::unordered_map<int, int> idToIdx;
    std::vector<int> idxToId;
    int nVtx = 0, nEdge = 0, k = 0;
};

bool loadGraph(const std::string& fname, Graph& g) {
    std::ifstream fin(fname);
    if (!fin) {
        std::cerr << "Failed to open " << fname << '\n';
        return false;
    }

    fin >> g.nVtx >> g.nEdge >> g.k;

    std::unordered_set<int> uniq;
    int u, v;
    auto pos = fin.tellg();

    while (fin >> u >> v)
        uniq.insert(u), uniq.insert(v);

    int idx = 0;
    for (auto id : uniq)
        g.idToIdx[id] = idx++, g.idxToId.push_back(id);

    fin.clear();
    fin.seekg(pos);

    g.adj.resize(uniq.size());
    while (fin >> u >> v) {
        int a = g.idToIdx[u], b = g.idToIdx[v];
        if (a != b) {
            auto& la = g.adj[a], &lb = g.adj[b];
            if (std::find(la.begin(), la.end(), b) == la.end()) {
                la.push_back(b);
                lb.push_back(a);
            }
        }
    }

    std::cout << "Graph loaded: " << uniq.size() << " vertices, "
              << g.nEdge << " edges, k = " << g.k << '\n';
    return true;
}

std::vector<int> calcCores(const Graph& g) {
    int n = g.adj.size();
    std::vector<int> deg(n), core(n);
    int maxDeg = 0;

    for (int i = 0; i < n; ++i)
        maxDeg = std::max(maxDeg, (deg[i] = core[i] = g.adj[i].size()));

    std::vector<std::vector<int>> bucket(maxDeg + 1);
    for (int i = 0; i < n; ++i)
        bucket[deg[i]].push_back(i);

    std::vector<bool> vis(n);
    for (int d = 0, left = n; d <= maxDeg && left; ++d) {
        while (!bucket[d].empty()) {
            int v = bucket[d].back(); bucket[d].pop_back();
            if (vis[v]) continue;
            vis[v] = true; --left;
            core[v] = d;
            for (int u : g.adj[v]) if (!vis[u] && deg[u] > d) {
                auto& bu = bucket[deg[u]];
                bu.erase(std::find(bu.begin(), bu.end(), u));
                bucket[--deg[u]].push_back(u);
            }
        }
    }
    return core;
}

void densestSubgraph(const Graph& g, std::vector<int>& best, double& bestDens) {
    const int n = g.adj.size(), k = g.k, ROUNDS = 100;
    std::vector<double> score(n, 0.0);
    auto core = calcCores(g);

    auto isClique = [&](const std::vector<int>& nodes) {
        for (int i = 0; i < nodes.size(); ++i)
            for (int j = i + 1; j < nodes.size(); ++j)
                if (std::find(g.adj[nodes[i]].begin(), g.adj[nodes[i]].end(), nodes[j]) == g.adj[nodes[i]].end())
                    return false;
        return true;
    };

    auto enumerate = [&](auto&& self, int depth, std::vector<int>& clique, std::vector<int>& cand, int start) -> void {
        if (depth == k) {
            int minV = *std::min_element(clique.begin(), clique.end(), [&](int a, int b) {
                return score[a] < score[b];
            });
            score[minV] += 1.0;
            return;
        }
        for (int i = start; i < cand.size(); ++i) {
            int v = cand[i];
            if (core[v] < k - 1) continue;
            bool good = true;
            for (int j = 0; j < depth; ++j)
                if (std::find(g.adj[clique[j]].begin(), g.adj[clique[j]].end(), v) == g.adj[clique[j]].end()) {
                    good = false;
                    break;
                }
            if (good) {
                clique[depth] = v;
                self(self, depth + 1, clique, cand, i + 1);
            }
        }
    };

    for (int r = 0; r < ROUNDS; ++r) {
        std::vector<int> cand(n), clique(k);
        for (int i = 0; i < n; ++i) cand[i] = i;
        enumerate(enumerate, 0, clique, cand, 0);
    }

    for (auto& s : score) s /= ROUNDS;

    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return score[a] > score[b]; });

    bestDens = 0.0;
    auto countCliques = [&](const std::vector<int>& sub) {
        int cnt = 0;
        auto dfs = [&](auto&& self, int d, std::vector<int>& clq, int idx) -> void {
            if (d == k) {
                ++cnt;
                return;
            }
            for (int i = idx; i < sub.size(); ++i) {
                bool ok = true;
                for (int j = 0; j < d; ++j)
                    if (std::find(g.adj[clq[j]].begin(), g.adj[clq[j]].end(), sub[i]) == g.adj[clq[j]].end()) {
                        ok = false;
                        break;
                    }
                if (ok) {
                    clq[d] = sub[i];
                    self(self, d + 1, clq, i + 1);
                }
            }
        };
        std::vector<int> clq(k);
        dfs(dfs, 0, clq, 0);
        return cnt;
    };

    for (int sz = k; sz <= n; ++sz) {
        std::vector<int> subset(order.begin(), order.begin() + sz);
        int cnt = countCliques(subset);
        if (cnt) {
            double dens = 1.0 * cnt / sz;
            if (dens > bestDens) {
                bestDens = dens;
                best = subset;
            }
        }
    }
}

int subgraphEdges(const Graph& g, const std::vector<int>& nodes) {
    std::unordered_set<int> s(nodes.begin(), nodes.end());
    int cnt = 0;
    for (int u : nodes)
        for (int v : g.adj[u])
            if (s.count(v) && u < v)
                ++cnt;
    return cnt;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    Graph g;
    if (!loadGraph(argv[1], g))
        return 1;

    std::vector<int> best;
    double bestDens = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::cout << "Running densest " << g.k << "-clique search...\n";

    densestSubgraph(g, best, bestDens);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Done in " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";

    int edges = subgraphEdges(g, best);

    std::cout << "\nRESULTS:\n";
    std::cout << "Vertices: " << best.size() << '\n';
    std::cout << "Nodes: ";
    for (auto v : best) std::cout << g.idxToId[v] << " ";
    std::cout << '\n';
    std::cout << "Edges: " << edges << '\n';
    std::cout << g.k << "-clique density: " << std::fixed << std::setprecision(6) << bestDens << '\n';

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "\nTotal time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    return 0;
}
