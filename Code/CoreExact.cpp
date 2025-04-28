#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <queue>
#include <string>
#include <set>
#include <cmath>
#include <limits>
#include <iomanip>
#include <chrono>

class Graph {
private:
    int n;
    std::unordered_map<int, std::unordered_set<int>> adjList;
    std::unordered_map<int, std::vector<std::vector<int>>> cliques;
    int h;

public:
    Graph(int vertices, int cliqueSize) : n(vertices), h(cliqueSize) {}

    void addEdge(int u, int v) {
        adjList[u].insert(v);
        adjList[v].insert(u);
    }

    void addClique(const std::vector<int>& clique) {
        for (int v : clique) {
            cliques[v].push_back(clique);
        }
    }

    const std::unordered_map<int, std::unordered_set<int>>& getAdjList() const {
        return adjList;
    }

    int getNumVertices() const {
        return n;
    }

    void find2Cliques() {
        for (const auto& [u, neighbors] : adjList) {
            for (int v : neighbors) {
                if (v > u) {
                    addClique({u, v});
                }
            }
        }
    }

    void findCliques() {
        if (h == 2) {
            find2Cliques();
        }
        if (h == 3) {
            findTriangles();
        } else if (h == 4) {
            find4Cliques();
        } else if (h == 5) {
            find5Cliques();
        } else if (h == 6) {
            find6Cliques();
        } else {
        }
    }

    void findTriangles() {
        for (const auto& [u, neighbors] : adjList) {
            for (int v : neighbors) {
                if (v > u) {
                    for (int w : adjList.at(v)) {
                        if (w > v && adjList.at(u).count(w)) {
                            addClique({u, v, w});
                        }
                    }
                }
            }
        }
    }

    void find4Cliques() {
        for (const auto& [u, neighbors1] : adjList) {
            for (int v : neighbors1) {
                if (v > u) {
                    for (int w : adjList.at(v)) {
                        if (w > v && adjList.at(u).count(w)) {
                            for (int x : adjList.at(w)) {
                                if (x > w && adjList.at(u).count(x) && adjList.at(v).count(x)) {
                                    addClique({u, v, w, x});
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void find5Cliques() {
        for (const auto& [u, neighbors1] : adjList) {
            for (int v : neighbors1) {
                if (v > u) {
                    for (int w : adjList.at(v)) {
                        if (w > v && adjList.at(u).count(w)) {
                            for (int x : adjList.at(w)) {
                                if (x > w && adjList.at(u).count(x) && adjList.at(v).count(x)) {
                                    for (int y : adjList.at(x)) {
                                        if (y > x && adjList.at(u).count(y) && adjList.at(v).count(y) && 
                                            adjList.at(w).count(y)) {
                                            addClique({u, v, w, x, y});
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void find6Cliques() {
        for (const auto& [u, neighbors1] : adjList) {
            for (int v : neighbors1) {
                if (v > u) {
                    for (int w : adjList.at(v)) {
                        if (w > v && adjList.at(u).count(w)) {
                            for (int x : adjList.at(w)) {
                                if (x > w && adjList.at(u).count(x) && adjList.at(v).count(x)) {
                                    for (int y : adjList.at(x)) {
                                        if (y > x && adjList.at(u).count(y) && adjList.at(v).count(y) && 
                                            adjList.at(w).count(y)) {
                                            for (int z : adjList.at(y)) {
                                                if (z > y && adjList.at(u).count(z) && adjList.at(v).count(z) && 
                                                    adjList.at(w).count(z) && adjList.at(x).count(z)) {
                                                    addClique({u, v, w, x, y, z});
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<int> computeCliqueDegrees() {
        std::vector<int> cliqueDegree(n + 1, 0);
        
        for (const auto& [v, cliqueList] : cliques) {
            cliqueDegree[v] = cliqueList.size();
        }
        
        return cliqueDegree;
    }

    std::vector<int> coreDecomposition() {
        std::vector<int> core(n + 1, 0);
        
        std::vector<int> cliqueDegree = computeCliqueDegrees();
        
        std::vector<int> vertices;
        for (const auto& [v, _] : adjList) {
            vertices.push_back(v);
        }
        
        std::sort(vertices.begin(), vertices.end(), 
                 [&cliqueDegree](int a, int b) {
                     return cliqueDegree[a] < cliqueDegree[b];
                 });
        
        std::unordered_set<int> remaining(vertices.begin(), vertices.end());
        
        while (!remaining.empty()) {
            auto minIt = std::min_element(remaining.begin(), remaining.end(),
                                        [&cliqueDegree](int a, int b) {
                                            return cliqueDegree[a] < cliqueDegree[b];
                                        });
            
            int v = *minIt;
            core[v] = cliqueDegree[v];
            
            remaining.erase(v);
            
            for (const auto& clique : cliques[v]) {
                for (int u : clique) {
                    if (remaining.count(u) && cliqueDegree[u] > cliqueDegree[v]) {
                        cliqueDegree[u]--;
                    }
                }
            }
        }
        
        return core;
    }

    std::vector<std::unordered_set<int>> findConnectedComponents(const std::vector<int>& core, int k) {
        std::vector<std::unordered_set<int>> components;
        std::unordered_set<int> visited;
        
        for (const auto& [v, _] : adjList) {
            if (core[v] >= k && visited.count(v) == 0) {
                std::unordered_set<int> component;
                std::queue<int> q;
                q.push(v);
                visited.insert(v);
                
                while (!q.empty()) {
                    int u = q.front();
                    q.pop();
                    component.insert(u);
                    
                    for (int w : adjList.at(u)) {
                        if (core[w] >= k && visited.count(w) == 0) {
                            q.push(w);
                            visited.insert(w);
                        }
                    }
                }
                
                components.push_back(component);
            }
        }
        
        return components;
    }

    double calculateCliqueSubgraphDensity(const std::unordered_set<int>& vertices) {
        if (vertices.empty()) return 0.0;
        
        int cliqueCount = 0;
        std::unordered_set<std::vector<int>, VectorHash> countedCliques;
        
        for (int v : vertices) {
            for (const auto& clique : cliques[v]) {
                bool allInSubgraph = true;
                for (int u : clique) {
                    if (vertices.count(u) == 0) {
                        allInSubgraph = false;
                        break;
                    }
                }
                
                if (allInSubgraph) {
                    std::vector<int> sortedClique = clique;
                    std::sort(sortedClique.begin(), sortedClique.end());
                    
                    if (countedCliques.count(sortedClique) == 0) {
                        countedCliques.insert(sortedClique);
                        cliqueCount++;
                    }
                }
            }
        }
        
        return static_cast<double>(cliqueCount) / vertices.size();
    }

    void buildFlowNetwork(const std::unordered_set<int>& vertices, int threshold, 
                         std::vector<std::vector<int>>& capacity, 
                         std::unordered_map<int, int>& vertexToIndex) {
        int n = vertices.size();
        int s = 0;
        int t = n + 1;
        
        capacity.assign(n + 2, std::vector<int>(n + 2, 0));
        
        int idx = 1;
        for (int v : vertices) {
            vertexToIndex[v] = idx++;
        }
        
        std::unordered_map<int, int> indexToVertex;
        for (const auto& [v, idx] : vertexToIndex) {
            indexToVertex[idx] = v;
        }
        
        for (int v : vertices) {
            int vIdx = vertexToIndex[v];
            int cliqueCount = 0;
            
            for (const auto& clique : cliques[v]) {
                bool allInSubgraph = true;
                for (int u : clique) {
                    if (vertices.count(u) == 0) {
                        allInSubgraph = false;
                        break;
                    }
                }
                
                if (allInSubgraph) {
                    cliqueCount++;
                }
            }
            
            if (cliqueCount > threshold) {
                capacity[s][vIdx] = cliqueCount - threshold;
            } else if (cliqueCount < threshold) {
                capacity[vIdx][t] = threshold - cliqueCount;
            }
            
            for (int u : adjList.at(v)) {
                if (vertices.count(u) && u > v) {
                    int uIdx = vertexToIndex[u];
                    
                    int sharedCliques = 0;
                    for (const auto& clique : cliques[v]) {
                        bool containsU = false;
                        bool allInSubgraph = true;
                        
                        for (int w : clique) {
                            if (w == u) {
                                containsU = true;
                            }
                            if (vertices.count(w) == 0) {
                                allInSubgraph = false;
                                break;
                            }
                        }
                        
                        if (containsU && allInSubgraph) {
                            sharedCliques++;
                        }
                    }
                    
                    capacity[vIdx][uIdx] = sharedCliques;
                    capacity[uIdx][vIdx] = sharedCliques;
                }
            }
        }
    }

    std::unordered_set<int> findMinCut(const std::vector<std::vector<int>>& capacity, 
                                     const std::unordered_map<int, int>& vertexToIndex,
                                     const std::unordered_set<int>& vertices) {
        int n = vertices.size();
        int s = 0;
        int t = n + 1;
        
        std::vector<std::vector<int>> residual = capacity;
        
        while (true) {
            std::vector<int> parent(n + 2, -1);
            std::queue<int> q;
            q.push(s);
            parent[s] = -2;
            
            while (!q.empty() && parent[t] == -1) {
                int u = q.front();
                q.pop();
                
                for (int v = 0; v <= n + 1; v++) {
                    if (parent[v] == -1 && residual[u][v] > 0) {
                        q.push(v);
                        parent[v] = u;
                    }
                }
            }
            
            if (parent[t] == -1) {
                break;
            }
            
            int minCapacity = std::numeric_limits<int>::max();
            for (int v = t; v != s; v = parent[v]) {
                int u = parent[v];
                minCapacity = std::min(minCapacity, residual[u][v]);
            }
            
            for (int v = t; v != s; v = parent[v]) {
                int u = parent[v];
                residual[u][v] -= minCapacity;
                residual[v][u] += minCapacity;
            }
        }
        
        std::unordered_set<int> visited;
        std::queue<int> q;
        q.push(s);
        visited.insert(s);
        
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            
            for (int v = 0; v <= n + 1; v++) {
                if (visited.count(v) == 0 && residual[u][v] > 0) {
                    q.push(v);
                    visited.insert(v);
                }
            }
        }
        
        std::unordered_set<int> sourceSet;
        std::unordered_map<int, int> indexToVertex;
        for (const auto& [v, idx] : vertexToIndex) {
            indexToVertex[idx] = v;
        }
        
        for (int idx : visited) {
            if (idx != s && idx != t) {
                sourceSet.insert(indexToVertex[idx]);
            }
        }
        
        return sourceSet;
    }

    std::pair<std::unordered_set<int>, double> coreExact() {
        std::vector<int> core = coreDecomposition();
        
        int kMax = 0;
        for (const auto& [v, _] : adjList) {
            kMax = std::max(kMax, core[v]);
        }
        
        int kPrime = kMax;
        
        int kDoublePrime = kPrime + 1;
        
        std::vector<std::unordered_set<int>> C;
        std::unordered_set<int> D;
        std::unordered_set<int> U;
        double l = 0;
        double u = kMax;
        
        C = findConnectedComponents(core, kPrime);
        
        double maxDensity = 0.0;
        
        for (const auto& component : C) {
            double density = calculateCliqueSubgraphDensity(component);
            
            if (l > kPrime) {
                continue;
            }
            
            std::vector<std::vector<int>> capacity;
            std::unordered_map<int, int> vertexToIndex;
            buildFlowNetwork(component, l, capacity, vertexToIndex);
            
            std::unordered_set<int> S = findMinCut(capacity, vertexToIndex, component);
            
            if (S.empty()) {
                continue;
            }
            
            U = S;
            while (u - l > 1.0 / component.size()) {
                double mid = (l + u) / 2;
                
                buildFlowNetwork(component, mid, capacity, vertexToIndex);
                
                S = findMinCut(capacity, vertexToIndex, component);
                
                if (S.empty()) {
                    u = mid;
                } else {
                    if (mid > l) {
                        U = S;
                    }
                    l = mid;
                }
            }
            
            density = calculateCliqueSubgraphDensity(U);
            if (density > maxDensity) {
                maxDensity = density;
                D = U;
            }
        }
        
        return {D, maxDensity};
    }

    struct VectorHash {
        size_t operator()(const std::vector<int>& v) const {
            size_t hash = v.size();
            for (auto i : v) {
                hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };
};

Graph readGraphFromFile(const std::string& filename, int h) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    
    int n, m;
    file >> n >> m;
    
    Graph graph(n, h);
    
    int u, v;
    while (file >> u >> v) {
        graph.addEdge(u, v);
    }
    
    file.close();
    return graph;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <filename> <h>" << std::endl;
        return 1;
    }
    
    std::string filename = argv[1];
    int h = std::stoi(argv[2]);
    
    std::cout << "h=" << h << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    Graph g = readGraphFromFile(filename, h);
    
    g.findCliques();
    
    auto [densestSubgraph, density] = g.coreExact();

    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Densest subgraph has " << densestSubgraph.size() << " vertices and " 
              << (densestSubgraph.size() * (densestSubgraph.size() - 1)) / 2 << " edges" << std::endl;
    std::cout << "h-Clique Density: " << std::fixed << std::setprecision(6) << density << std::endl;
    
    std::cout << "Vertices: ";
    std::vector<int> sortedVertices(densestSubgraph.begin(), densestSubgraph.end());
    std::sort(sortedVertices.begin(), sortedVertices.end());
    for (int v : sortedVertices) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    std::cout << "\nExecution time: " << duration.count() << " milliseconds" << std::endl;
    
    return 0;
}
