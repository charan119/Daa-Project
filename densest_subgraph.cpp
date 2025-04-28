#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <string>
#include <sstream>
#include <chrono>
#include <limits>
#include <cmath>

using namespace std;
using namespace chrono;

class Graph
{
public:
    int number_of_nodes;
    vector<vector<int>> adjacency_list;
    vector<int> node_degrees;

    Graph(int nodes)
    {
        number_of_nodes = nodes;
        adjacency_list.resize(nodes);
        node_degrees.resize(nodes, 0);
    }

    void resize_graph(int new_size)
    {
        number_of_nodes = new_size;
        adjacency_list.resize(new_size);
        node_degrees.resize(new_size, 0);
    }

    bool check_edge_exists(int node1, vector<int> neighbors, int node2)
    {
        for (int i = 0; i < neighbors.size(); i++)
        {
            if (neighbors[i] == node2)
            {
                return true;
            }
        }
        return false;
    }

    void add_single_edge(int from_node, int to_node)
    {
        if (!check_edge_exists(from_node, adjacency_list[from_node], to_node))
        {
            adjacency_list[from_node].push_back(to_node);
            node_degrees[from_node]++;
        }
    }

    void add_edge_both_ways(int node_u, int node_v)
    {
        int max_node_number = max(node_u, node_v);
        if (max_node_number >= number_of_nodes)
        {
            resize_graph(max_node_number + 1);
        }
        add_single_edge(node_u, node_v);
        add_single_edge(node_v, node_u);
    }

    bool open_file_stream(ifstream &file, const string &file_name)
    {
        file.open(file_name);
        if (!file.is_open())
        {
            cerr << "Error: Cannot open " << file_name << endl;
            exit(1);
        }
        return true;
    }

    bool check_matrix_market(string first_line)
    {
        if (first_line.find("MatrixMarket") != string::npos)
        {
            return true;
        }
        return false;
    }

    void skip_matrix_market_comments(ifstream &file)
    {
        string line;
        while (getline(file, line))
        {
            if (line[0] != '%')
            {
                file.seekg(-line.length() - 1, ios_base::cur);
                break;
            }
        }
    }

    void reset_file_pointer(ifstream &file)
    {
        file.clear();
        file.seekg(0);
    }

    void read_matrix_market_header(string line, int &rows, int &cols, int &entries)
    {
        istringstream iss(line);
        iss >> rows >> cols >> entries;
    }

    void initialize_graph_for_matrix_market(int rows, int cols)
    {
        number_of_nodes = max(rows, cols);
        adjacency_list.assign(number_of_nodes, vector<int>());
        node_degrees.assign(number_of_nodes, 0);
    }

    void process_matrix_market_edge(ifstream &file)
    {
        string line;
        while (getline(file, line))
        {
            if (line.empty() || line[0] == '%')
            {
                continue;
            }
            istringstream iss(line);
            int u, v;
            iss >> u >> v;
            if (u != v)
            {
                add_edge_both_ways(u - 1, v - 1);
            }
        }
    }

    void process_text_edge(ifstream &file, unordered_set<int> &vertices)
    {
        string line;
        while (getline(file, line))
        {
            if (line.empty() || line[0] == '%' || line[0] == '#')
            {
                continue;
            }
            istringstream iss(line);
            int u, v;
            iss >> u >> v;
            vertices.insert(u);
            vertices.insert(v);
            if (u != v)
            {
                add_edge_both_ways(u, v);
            }
        }
    }

    void clean_adjacency_lists()
    {
        for (int i = 0; i < number_of_nodes; i++)
        {
            sort(adjacency_list[i].begin(), adjacency_list[i].end());
            vector<int> unique_list;
            for (int j = 0; j < adjacency_list[i].size(); j++)
            {
                if (j == 0 || adjacency_list[i][j] != adjacency_list[i][j - 1])
                {
                    unique_list.push_back(adjacency_list[i][j]);
                }
            }
            adjacency_list[i] = unique_list;
            node_degrees[i] = adjacency_list[i].size();
        }
    }

    void load_from_file(const string &filename)
    {
        ifstream file;
        open_file_stream(file, filename);
        string line;
        bool is_matrix_market = false;
        if (getline(file, line))
        {
            is_matrix_market = check_matrix_market(line);
        }
        if (is_matrix_market)
        {
            skip_matrix_market_comments(file);
            getline(file, line);
            int rows, cols, entries;
            read_matrix_market_header(line, rows, cols, entries);
            initialize_graph_for_matrix_market(rows, cols);
            process_matrix_market_edge(file);
        }
        else
        {
            reset_file_pointer(file);
            number_of_nodes = 0;
            unordered_set<int> vertices;
            process_text_edge(file, vertices);
            number_of_nodes = max(number_of_nodes, (int)vertices.size());
        }
        clean_adjacency_lists();
        file.close();
    }
};

class FlowNetwork
{
public:
    struct Edge
    {
        int destination;
        double capacity, flow;
        int reverse_index;
        Edge(int dest, double cap, int rev)
        {
            destination = dest;
            capacity = cap;
            flow = 0;
            reverse_index = rev;
        }
    };

    vector<vector<Edge>> edge_list;
    vector<int> level_list, pointer_list;
    int node_count;

    FlowNetwork(int nodes)
    {
        node_count = nodes;
        edge_list.resize(nodes);
        level_list.resize(nodes);
        pointer_list.resize(nodes);
    }

    void add_one_edge(int start, int end, double cap)
    {
        edge_list[start].push_back(Edge(end, cap, edge_list[end].size()));
        edge_list[end].push_back(Edge(start, 0, edge_list[start].size() - 1));
    }

    void clear_levels()
    {
        for (int i = 0; i < node_count; i++)
        {
            level_list[i] = -1;
        }
    }

    void set_source_level(int source)
    {
        level_list[source] = 0;
    }

    bool process_node_bfs(int node, queue<int> &q)
    {
        q.pop();
        for (size_t i = 0; i < edge_list[node].size(); i++)
        {
            Edge &e = edge_list[node][i];
            if (level_list[e.destination] == -1 && e.capacity - e.flow > 1e-9)
            {
                level_list[e.destination] = level_list[node] + 1;
                q.push(e.destination);
            }
        }
        return !q.empty();
    }

    bool run_bfs(int source, int sink)
    {
        clear_levels();
        set_source_level(source);
        queue<int> q;
        q.push(source);
        while (!q.empty() && level_list[sink] == -1)
        {
            int current_node = q.front();
            process_node_bfs(current_node, q);
        }
        return level_list[sink] != -1;
    }

    double dfs_flow(int node, int sink, double flow)
    {
        if (node == sink)
        {
            return flow;
        }
        for (int &i = pointer_list[node]; i < (int)edge_list[node].size(); i++)
        {
            Edge &e = edge_list[node][i];
            if (level_list[node] + 1 == level_list[e.destination] && e.capacity - e.flow > 1e-9)
            {
                double pushed = dfs_flow(e.destination, sink, min(flow, e.capacity - e.flow));
                if (pushed > 1e-9)
                {
                    e.flow += pushed;
                    edge_list[e.destination][e.reverse_index].flow -= pushed;
                    return pushed;
                }
            }
        }
        return 0;
    }

    void reset_pointers()
    {
        for (int i = 0; i < node_count; i++)
        {
            pointer_list[i] = 0;
        }
    }

    double compute_max_flow(int source, int sink)
    {
        double total_flow = 0;
        while (run_bfs(source, sink))
        {
            reset_pointers();
            double pushed_flow;
            while ((pushed_flow = dfs_flow(source, sink, numeric_limits<double>::max())) > 1e-9)
            {
                total_flow += pushed_flow;
            }
        }
        return total_flow;
    }

    void mark_visited(int node, vector<bool> &visited, queue<int> &q)
    {
        visited[node] = true;
        q.push(node);
    }

    void explore_cut(int node, vector<bool> &visited, queue<int> &q)
    {
        q.pop();
        for (size_t i = 0; i < edge_list[node].size(); i++)
        {
            Edge &e = edge_list[node][i];
            if (!visited[e.destination] && e.capacity - e.flow > 1e-9)
            {
                mark_visited(e.destination, visited, q);
            }
        }
    }

    vector<int> find_minimum_cut(int source)
    {
        vector<bool> visited(node_count, false);
        queue<int> q;
        mark_visited(source, visited, q);
        while (!q.empty())
        {
            int current_node = q.front();
            explore_cut(current_node, visited, q);
        }
        vector<int> cut_nodes;
        for (int i = 1; i < node_count - 1; i++)
        {
            if (visited[i])
            {
                cut_nodes.push_back(i - 1);
            }
        }
        return cut_nodes;
    }
};

vector<vector<int>> find_two_cliques(const Graph &graph)
{
    vector<vector<int>> clique_list;
    for (int u = 0; u < graph.number_of_nodes; u++)
    {
        for (int v : graph.adjacency_list[u])
        {
            if (u < v)
            {
                clique_list.push_back({u, v});
            }
        }
    }
    return clique_list;
}

vector<unordered_set<int>> build_neighbor_sets(const Graph &graph)
{
    vector<unordered_set<int>> neighbors(graph.number_of_nodes);
    for (int u = 0; u < graph.number_of_nodes; u++)
    {
        for (int v : graph.adjacency_list[u])
        {
            neighbors[u].insert(v);
        }
    }
    return neighbors;
}

vector<vector<int>> find_three_cliques(const Graph &graph)
{
    vector<vector<int>> clique_list;
    vector<unordered_set<int>> neighbors = build_neighbor_sets(graph);
    for (int u = 0; u < graph.number_of_nodes; u++)
    {
        for (int v : graph.adjacency_list[u])
        {
            if (v > u)
            {
                for (int w : graph.adjacency_list[v])
                {
                    if (w > v && neighbors[u].count(w))
                    {
                        clique_list.push_back({u, v, w});
                    }
                }
            }
        }
    }
    return clique_list;
}

void backtrack_cliques(int pos, vector<int> &current, vector<bool> &used, const Graph &graph, int k, vector<vector<int>> &cliques)
{
    if (current.size() == k)
    {
        cliques.push_back(current);
        return;
    }
    if (current.size() + (graph.number_of_nodes - pos) < k)
    {
        return;
    }
    for (int i = pos; i < graph.number_of_nodes; i++)
    {
        if (!used[i])
        {
            bool is_valid = true;
            for (int u : current)
            {
                bool found = false;
                for (int v : graph.adjacency_list[u])
                {
                    if (v == i)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    is_valid = false;
                    break;
                }
            }
            if (is_valid)
            {
                current.push_back(i);
                used[i] = true;
                backtrack_cliques(i + 1, current, used, graph, k, cliques);
                current.pop_back();
                used[i] = false;
            }
        }
    }
}

vector<vector<int>> enumerate_k_cliques(const Graph &graph, int k_value)
{
    vector<vector<int>> clique_list;
    if (k_value == 2)
    {
        clique_list = find_two_cliques(graph);
        return clique_list;
    }
    if (k_value == 3)
    {
        clique_list = find_three_cliques(graph);
        return clique_list;
    }
    vector<int> current_clique;
    vector<bool> used_nodes(graph.number_of_nodes, false);
    backtrack_cliques(0, current_clique, used_nodes, graph, k_value, clique_list);
    return clique_list;
}

double calculate_clique_density(const vector<vector<int>> &cliques, const vector<int> &subgraph)
{
    if (subgraph.empty())
    {
        return 0.0;
    }
    unordered_set<int> subgraph_set;
    for (int v : subgraph)
    {
        subgraph_set.insert(v);
    }
    int clique_count = 0;
    for (const auto &clique : cliques)
    {
        bool all_vertices_in = true;
        for (int v : clique)
        {
            if (subgraph_set.find(v) == subgraph_set.end())
            {
                all_vertices_in = false;
                break;
            }
        }
        if (all_vertices_in)
        {
            clique_count++;
        }
    }
    return static_cast<double>(clique_count) / subgraph.size();
}

void setup_flow_network_edges(FlowNetwork &fn, const Graph &graph, int source, int sink, double alpha)
{
    for (int i = 0; i < graph.number_of_nodes; i++)
    {
        fn.add_one_edge(source, i + 1, graph.adjacency_list[i].size());
    }
    for (int i = 0; i < graph.number_of_nodes; i++)
    {
        fn.add_one_edge(i + 1, sink, 2 * alpha);
    }
    for (int i = 0; i < graph.number_of_nodes; i++)
    {
        for (int j : graph.adjacency_list[i])
        {
            if (i < j)
            {
                fn.add_one_edge(i + 1, j + 1, 1);
                fn.add_one_edge(j + 1, i + 1, 1);
            }
        }
    }
}

vector<int> compute_edge_density(const Graph resaltan, double alpha_value, vector<int> &best_subgraph, double &best_density_value)
{
    int source_node = 0;
    int sink_node = resaltan.number_of_nodes + 1;
    FlowNetwork flow_network(resaltan.number_of_nodes + 2);
    setup_flow_network_edges(flow_network, resaltan, source_node, sink_node, alpha_value);
    double flow_value = flow_network.compute_max_flow(source_node, sink_node);
    double potential_value = 2.0 * resaltan.number_of_nodes * alpha_value;
    if (flow_value < potential_value)
    {
        vector<int> subgraph = flow_network.find_minimum_cut(source_node);
        int edge_count = 0;
        unordered_set<int> subgraph_set(subgraph.begin(), subgraph.end());
        for (int u : subgraph)
        {
            for (int v : resaltan.adjacency_list[u])
            {
                if (subgraph_set.count(v))
                {
                    edge_count++;
                }
            }
        }
        edge_count /= 2;
        double density = subgraph.empty() ? 0.0 : static_cast<double>(edge_count) / subgraph.size();
        if (density > best_density_value)
        {
            best_density_value = density;
            best_subgraph = subgraph;
        }
        return subgraph;
    }
    return vector<int>();
}

void setup_clique_flow_network(FlowNetwork &fn, const Graph &graph, const vector<vector<int>> &h_minus_1_cliques, int source, int sink, double alpha)
{
    for (int i = 0; i < graph.number_of_nodes; i++)
    {
        fn.add_one_edge(source, i + 1, h * alpha);
    }
    for (size_t i = 0; i < h_minus_1_cliques.size(); i++)
    {
        fn.add_one_edge(graph.number_of_nodes + i + 1, sink, 1);
    }
    for (int i = 0; i < graph.number_of_nodes; i++)
    {
        for (size_t j = 0; j < h_minus_1_cliques.size(); j++)
        {
            const auto &clique = h_minus_1_cliques[j];
            bool vertex_in_clique = false;
            for (int v : clique)
            {
                if (v == i)
                {
                    vertex_in_clique = true;
                    break;
                }
            }
            if (vertex_in_clique)
            {
                continue;
            }
            bool can_extend = true;
            for (int v : clique)
            {
                bool connected = false;
                for (int neighbor : graph.adjacency_list[i])
                {
                    if (neighbor == v)
                    {
                        connected = true;
                        break;
                    }
                }
                if (!connected)
                {
                    can_extend = false;
                    break;
                }
            }
            if (can_extend)
            {
                fn.add_one_edge(i + 1, graph.number_of_nodes + j + 1, 1);
            }
        }
    }
}

vector<int> compute_clique_density(const Graph &graph, const vector<vector<int>> &cliques, double alpha, int h, const vector<vector<int>> &h_minus_1_cliques, vector<int> &best_subgraph, double &best_density)
{
    int source = 0;
    int sink = graph.number_of_nodes + h_minus_1_cliques.size() + 1;
    FlowNetwork fn(graph.number_of_nodes + h_minus_1_cliques.size() + 2);
    setup_clique_flow_network(fn, graph, h_minus_1_cliques, source, sink, alpha);
    double flow = fn.compute_max_flow(source, sink);
    if (flow < h_minus_1_cliques.size())
    {
        vector<int> subgraph = fn.find_minimum_cut(source);
        double density = calculate_clique_density(cliques, subgraph);
        if (density > best_density)
        {
            best_density = density;
            best_subgraph = subgraph;
        }
        return subgraph;
    }
    return vector<int>();
}

pair<vector<int>, double> run_algorithm_one(const Graph &graph, int h_value, const vector<vector<int>> &clique_list)
{
    vector<int> best_subgraph;
    double best_density_value = 0;
    if (h_value == 2)
    {
        double left = 0;
        double right = graph.number_of_nodes;
        for (int i = 0; i < 30; i++)
        {
            double alpha_value = (left + right) / 2.0;
            vector<int> subgraph = compute_edge_density(graph, alpha_value, best_subgraph, best_density_value);
            if (!subgraph.empty())
            {
                right = alpha_value;
            }
            else
            {
                left = alpha_value;
            }
        }
    }
    else
    {
        vector<vector<int>> h_minus_1_cliques = enumerate_k_cliques(graph, h_value - 1);
        double left = 0;
        double right = clique_list.size();
        for (int i = 0; i < 30; i++)
        {
            double alpha = (left + right) / 2.0;
            vector<int> subgraph = compute_clique_density(graph, clique_list, alpha, h_value, h_minus_1_cliques, best_subgraph, best_density_value);
            if (!subgraph.empty())
            {
                right = alpha;
            }
            else
            {
                left = alpha;
            }
        }
    }
    return make_pair(best_subgraph, best_density_value);
}

vector<int> compute_clique_degrees(const Graph &graph, const vector<vector<int>> &cliques)
{
    vector<int> degrees(graph.number_of_nodes, 0);
    for (const auto &clique : cliques)
    {
        for (int v : clique)
        {
            degrees[v]++;
        }
    }
    return degrees;
}

vector<int> initialize_subgraph(const Graph &graph)
{
    vector<int> subgraph;
    for (int i = 0; i < graph.number_of_nodes; i++)
    {
        subgraph.push_back(i);
    }
    return subgraph;
}

void find_minimum_degree_vertex(const vector<int> &degrees, const vector<int> &subgraph, int &min_degree, int &min_vertex, int &min_index)
{
    min_degree = numeric_limits<int>::max();
    min_vertex = -1;
    min_index = -1;
    for (size_t i = 0; i < subgraph.size(); i++)
    {
        int v = subgraph[i];
        if (degrees[v] < min_degree)
        {
            min_degree = degrees[v];
            min_vertex = v;
            min_index = i;
        }
    }
}

void update_clique_degrees(const vector<vector<int>> &cliques, const vector<int> &subgraph, vector<int> &degrees)
{
    unordered_set<int> subgraph_set(subgraph.begin(), subgraph.end());
    fill(degrees.begin(), degrees.end(), 0);
    for (const auto &clique : cliques)
    {
        bool all_in = true;
        for (int v : clique)
        {
            if (subgraph_set.find(v) == subgraph_set.end())
            {
                all_in = false;
                break;
            }
        }
        if (all_in)
        {
            for (int v : clique)
            {
                degrees[v]++;
            }
        }
    }
}

pair<vector<int>, double> run_algorithm_four(const Graph &graph, int h_value, const vector<vector<int>> &clique_list)
{
    vector<int> degrees = compute_clique_degrees(graph, clique_list);
    vector<int> subgraph = initialize_subgraph(graph);
    vector<int> best_subgraph = subgraph;
    double best_density = calculate_clique_density(clique_list, subgraph);
    while (!subgraph.empty())
    {
        int min_degree, min_vertex, min_index;
        find_minimum_degree_vertex(degrees, subgraph, min_degree, min_vertex, min_index);
        if (min_index == -1)
        {
            break;
        }
        subgraph.erase(subgraph.begin() + min_index);
        update_clique_degrees(clique_list, subgraph, degrees);
        if (!subgraph.empty())
        {
            double density = calculate_clique_density(clique_list, subgraph);
            if (density > best_density)
            {
                best_density = density;
                best_subgraph = subgraph;
            }
        }
    }
    return make_pair(best_subgraph, best_density);
}

void process_dataset(const string &dataset, const string &name)
{
    Graph graph(0);
    try
    {
        graph.load_from_file(dataset);
        cout << "\nProcessing " << name << " (Vertices: " << graph.number_of_nodes << ")\n";
        for (int h = 2; h <= 6; h++)
        {
            cout << "h = " << h << ":\n";
            auto start_time = high_resolution_clock::now();
            vector<vector<int>> cliques = enumerate_k_cliques(graph, h);
            auto end_time = high_resolution_clock::now();
            auto enum_time = duration_cast<milliseconds>(end_time - start_time).count();
            if (cliques.empty() && h > 2)
            {
                cout << "  No " << h << "-cliques found. Skipping.\n";
                continue;
            }
            start_time = high_resolution_clock::now();
            pair<vector<int>, double> result1 = run_algorithm_one(graph, h, cliques);
            vector<int> subgraph1 = result1.first;
            double density1 = result1.second;
            end_time = high_resolution_clock::now();
            auto time1 = duration_cast<milliseconds>(end_time - start_time).count();
            cout << "  Algorithm-1: Density = " << density1 << ", Time = " << time1 << "ms, Size = " << subgraph1.size() << "\n";
            start_time = high_resolution_clock::now();
            pair<vector<int>, double> result4 = run_algorithm_four(graph, h, cliques);
            vector<int> subgraph4 = result4.first;
            double density4 = result4.second;
            end_time = high_resolution_clock::now();
            auto time4 = duration_cast<milliseconds>(end_time - start_time).count();
            cout << "  Algorithm-4: Density = " << density4 << ", Time = " << time4 << "ms, Size = " << subgraph4.size() << "\n";
        }
    }
    catch (const exception &e)
    {
        cerr << "Error processing " << dataset << ": " << e.what() << endl;
    }
}

int main()
{
    vector<string> dataset_files = {
        "bio-yeast.txt",
        "ca-netscience.txt",
        "ca-HepTh.txt",
        "as-caida20040105.txt"};
    vector<string> dataset_names = {
        "Bio-Yeast",
        "Ca-Netscience",
        "Ca-HepTh",
        "As-Caida"};
    for (size_t i = 0; i < dataset_files.size(); i++)
    {
        process_dataset(dataset_files[i], dataset_names[i]);
    }
    return 0;
}