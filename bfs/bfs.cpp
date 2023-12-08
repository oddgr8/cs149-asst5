#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// #define VERBOSE 1

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{

    for (int i=0; i<frontier->count; i++) {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER) {
                distances[outgoing] = distances[node] + 1;
                int index = new_frontier->count++;
                new_frontier->vertices[index] = outgoing;
            }
        }
    }
}

void parallel_top_down_step(Graph g, vertex_set *frontier, vertex_set *new_frontier, int *distances){
    #pragma omp for
    for (int i = 0; i < frontier->count; i++){
        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        int nodes_to_add[end_edge - start_edge];
        int count = 0;

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++){
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] != NOT_VISITED_MARKER)
                continue;

            if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1)){
                nodes_to_add[count++] = outgoing;
            }
        }

        if (count > 0){
            int offset = __sync_fetch_and_add(&new_frontier->count, count);
            for (int i = 0; i < count; i++){
                new_frontier->vertices[offset + i] = nodes_to_add[i];
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;


    #pragma omp parallel
    {
        // initialize all nodes to NOT_VISITED
        #pragma omp for
        for (int i=0; i<graph->num_nodes; i++)
            sol->distances[i] = NOT_VISITED_MARKER;

        #pragma omp single
        {
            // setup frontier with the root node
            frontier->vertices[frontier->count++] = ROOT_NODE_ID;
            sol->distances[ROOT_NODE_ID] = 0;
        }

        while (frontier->count != 0){

#ifdef VERBOSE
            double start_time = CycleTimer::currentSeconds();
#endif
            #pragma omp single
            vertex_set_clear(new_frontier);

            // parallel_top_down_step(graph, frontier, new_frontier, sol->distances);
            #pragma omp for
            for (int i = 0; i < frontier->count; i++){
                int node = frontier->vertices[i];

                int start_edge = graph->outgoing_starts[node];
                int end_edge = (node == graph->num_nodes - 1)
                                   ? graph->num_edges
                                   : graph->outgoing_starts[node + 1];

                int nodes_to_add[end_edge - start_edge];
                int count = 0;

                // attempt to add all neighbors to the new frontier
                for (int neighbor = start_edge; neighbor < end_edge; neighbor++){
                    int outgoing = graph->outgoing_edges[neighbor];

                    if (sol->distances[outgoing] != NOT_VISITED_MARKER)
                        continue;

                    if (__sync_bool_compare_and_swap(&sol->distances[outgoing], NOT_VISITED_MARKER, sol->distances[node] + 1)){
                        nodes_to_add[count++] = outgoing;
                    }
                }

                if (count > 0){
                    int offset = __sync_fetch_and_add(&new_frontier->count, count);
                    for (int i = 0; i < count; i++){
                        new_frontier->vertices[offset + i] = nodes_to_add[i];
                    }
                }
            }

#ifdef VERBOSE
            double end_time = CycleTimer::currentSeconds();
            #pragma omp single
            printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

            #pragma omp single
            {
                // swap pointers
                vertex_set *tmp = frontier;
                frontier = new_frontier;
                new_frontier = tmp;
            }
        }
    }
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

//     vertex_set list1;
//     vertex_set list2;
//     vertex_set_init(&list1, graph->num_nodes);
//     vertex_set_init(&list2, graph->num_nodes);

//     vertex_set *frontier = &list1;
//     vertex_set *new_frontier = &list2;

//     // initialize all nodes to NOT_VISITED
//     for (int i = 0; i < graph->num_nodes; i++)
//         sol->distances[i] = NOT_VISITED_MARKER;

//     // setup frontier with the root node
//     frontier->vertices[frontier->count++] = ROOT_NODE_ID;
//     sol->distances[ROOT_NODE_ID] = 0;

//     while (frontier->count != 0){

// #ifdef VERBOSE
//         double start_time = CycleTimer::currentSeconds();
// #endif

//         vertex_set_clear(new_frontier);

//         #pragma omp parallel for
//         for(int node = 0; node < graph->num_nodes; node++){
//             if(sol->distances[node] != NOT_VISITED_MARKER)
//                 continue;

//             for(const Vertex* neighbor = incoming_begin(graph, node); neighbor < incoming_end(graph, node); neighbor++){
//                 if(sol->distances[*neighbor] != NOT_VISITED_MARKER){
//                     bool found = false;
//                     for(int i=0; i<frontier->count; i++){
//                         if(frontier->vertices[i] == *neighbor){
//                             found = true;
//                             break;
//                         }
//                     }
//                     if(found){
//                         #pragma omp critical
//                         {
//                             new_frontier->vertices[new_frontier->count++] = node;
//                         }
//                         sol->distances[node] = sol->distances[*neighbor] + 1;
//                         break;
//                     }
//                 }
//             }    
//         }

// #ifdef VERBOSE
//         double end_time = CycleTimer::currentSeconds();
//         printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
// #endif

//         vertex_set *tmp = frontier;
//         frontier = new_frontier;
//         new_frontier = tmp;
//     }
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
