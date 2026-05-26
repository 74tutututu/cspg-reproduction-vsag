
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <atomic>

#include "../monitor/monitor.h"
#include "./eval_case.h"

namespace vsag::eval {

class SearchEvalCase : public EvalCase {
public:
    SearchEvalCase(const std::string& dataset_path,
                   const std::string& index_path,
                   vsag::IndexPtr index,
                   EvalConfig config);

    ~SearchEvalCase() override = default;

    JsonType
    Run() override;

private:
    enum SearchType {
        KNN,
        RANGE,
        KNN_FILTER,
        RANGE_FILTER,
    };

    void
    init_monitor();

    void
    init_latency_monitor();

    void
    init_recall_monitor();

    void
    init_memory_monitor();

    void
    deserialize(std::ifstream& infile);

    void
    do_knn_search();

    void
    do_range_search();

    void
    do_knn_filter_search();

    void
    do_range_filter_search();

    void
    collect_search_stats(const vsag::DatasetPtr& result);

    JsonType
    process_result();

private:
    std::vector<MonitorPtr> monitors_{};

    SearchType search_type_{SearchType::KNN};

    EvalConfig config_;

    std::atomic<uint64_t> total_dist_cmp_{0};
    std::atomic<uint64_t> total_hops_{0};
    std::atomic<uint64_t> total_stat_samples_{0};
    std::atomic<uint64_t> total_cspg_phase1_dist_cmp_{0};
    std::atomic<uint64_t> total_cspg_phase1_hops_{0};
    std::atomic<uint64_t> total_cspg_stage2_dist_cmp_{0};
    std::atomic<uint64_t> total_cspg_stage2_hops_{0};
    std::atomic<uint64_t> total_cspg_stage2_local_dist_cmp_{0};
    std::atomic<uint64_t> total_cspg_stage2_cross_partition_dist_cmp_{0};
    std::atomic<uint64_t> total_cspg_stage2_local_routing_dist_cmp_{0};
    std::atomic<uint64_t> total_cspg_stage2_local_nonrouting_dist_cmp_{0};
    std::atomic<uint64_t> total_cspg_stage2_routing_pops_{0};
    std::atomic<uint64_t> total_cspg_stage2_nonrouting_pops_{0};
    std::atomic<uint64_t> total_cspg_stage2_routing_useless_pops_{0};
    std::atomic<uint64_t> total_cspg_stage2_nonrouting_useless_pops_{0};
    std::atomic<uint64_t> total_cspg_stage2_routing_fanout_attempts_{0};
    std::atomic<uint64_t> total_cspg_stage2_routing_fanout_enqueues_{0};
    std::atomic<uint64_t> total_cspg_stage2_routing_fanout_skipped_unexpandable_{0};
    std::atomic<uint64_t> total_cspg_stage2_routing_fanout_skipped_no_unvisited_{0};
    std::atomic<uint64_t> total_cspg_stage2_routing_fanout_skipped_duplicate_state_{0};
    std::atomic<uint64_t> total_cspg_stage2_routing_fanout_skipped_frontier_reject_{0};
    std::atomic<uint64_t> total_cspg_stage2_routing_fanout_skipped_by_bound_{0};
};
}  // namespace vsag::eval
