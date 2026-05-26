
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

#include "./search_eval_case.h"

#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../monitor/latency_monitor.h"
#include "../monitor/memory_peak_monitor.h"
#include "../monitor/recall_monitor.h"
#include "typing.h"
#include "vsag/filter.h"
#include "vsag_exception.h"

namespace vsag::eval {

class FilterObj : public vsag::Filter {
public:
    FilterObj(const std::shared_ptr<int64_t[]>& train_labels, int64_t test_label, float valid_ratio)
        : train_labels_(train_labels), test_label_(test_label), valid_ratio_(valid_ratio) {
    }

    bool
    CheckValid(int64_t id) const override {
        return train_labels_[id] == test_label_;
    }

    float
    ValidRatio() const override {
        return valid_ratio_;
    }

private:
    const std::shared_ptr<int64_t[]>& train_labels_;
    int64_t test_label_;
    float valid_ratio_;
};

SearchEvalCase::SearchEvalCase(const std::string& dataset_path,
                               const std::string& index_path,
                               vsag::IndexPtr index,
                               EvalConfig config)
    : EvalCase(dataset_path, index_path, index), config_(std::move(config)) {
    auto search_mode = config_.search_mode;
    if (search_mode == "knn") {
        this->search_type_ = SearchType::KNN;
    } else if (search_mode == "range") {
        this->search_type_ = SearchType::RANGE;
    } else if (search_mode == "knn_filter") {
        this->search_type_ = SearchType::KNN_FILTER;
    } else if (search_mode == "range_filter") {
        this->search_type_ = SearchType::RANGE_FILTER;
    }
    this->init_monitor();
}

void
SearchEvalCase::init_monitor() {
    this->init_latency_monitor();
    this->init_recall_monitor();
    this->init_memory_monitor();
}

void
SearchEvalCase::init_latency_monitor() {
    if (config_.enable_latency or config_.enable_tps or config_.enable_percent_latency) {
        auto latency_monitor =
            std::make_shared<LatencyMonitor>(this->dataset_ptr_->GetNumberOfQuery());
        if (config_.enable_qps) {
            latency_monitor->SetMetrics("qps");
        }
        if (config_.enable_latency) {
            latency_monitor->SetMetrics("avg_latency");
        }
        if (config_.enable_percent_latency) {
            latency_monitor->SetMetrics("percent_latency");
        }
        this->monitors_.emplace_back(std::move(latency_monitor));
    }
}

void
SearchEvalCase::init_recall_monitor() {
    if (config_.enable_recall or config_.enable_percent_recall) {
        auto recall_monitor =
            std::make_shared<RecallMonitor>(this->dataset_ptr_->GetNumberOfQuery());
        if (config_.enable_recall) {
            recall_monitor->SetMetrics("avg_recall");
        }
        if (config_.enable_percent_recall) {
            recall_monitor->SetMetrics("percent_recall");
        }
        this->monitors_.emplace_back(std::move(recall_monitor));
    }
}

void
SearchEvalCase::init_memory_monitor() {
    if (config_.enable_memory) {
        auto memory_peak_monitor = std::make_shared<MemoryPeakMonitor>("search");
        this->monitors_.emplace_back(std::move(memory_peak_monitor));
    }
}

JsonType
SearchEvalCase::Run() {
    std::ifstream infile(this->index_path_, std::ios::binary);
    this->deserialize(infile);
    switch (this->search_type_) {
        case KNN:
            this->do_knn_search();
            break;
        case RANGE:
            this->do_range_search();
            break;
        case KNN_FILTER:
            this->do_knn_filter_search();
            break;
        case RANGE_FILTER:
            this->do_range_filter_search();
            break;
    }
    auto result = this->process_result();
    if (config_.delete_index_after_search) {
        std::remove(this->index_path_.c_str());
    }
    return result;
}

void
SearchEvalCase::deserialize(std::ifstream& infile) {
    this->index_->Deserialize(infile);
}

void
SearchEvalCase::collect_search_stats(const vsag::DatasetPtr& result) {
    if (result == nullptr) {
        return;
    }

    if (!config_.enable_cspg_stats) {
        auto stats = result->GetStatistics({"dist_cmp", "hops"});
        if (stats.size() == 2 && !stats[0].empty() && !stats[1].empty()) {
            this->total_dist_cmp_.fetch_add(std::stoull(stats[0]), std::memory_order_relaxed);
            this->total_hops_.fetch_add(std::stoull(stats[1]), std::memory_order_relaxed);
            this->total_stat_samples_.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }

    auto stats = result->GetStatistics({"dist_cmp",
                                        "hops",
                                        "cspg_phase1_dist_cmp",
                                        "cspg_phase1_hops",
                                        "cspg_stage2_dist_cmp",
                                        "cspg_stage2_hops",
                                        "cspg_stage2_local_dist_cmp",
                                        "cspg_stage2_cross_partition_dist_cmp",
                                        "cspg_stage2_local_routing_dist_cmp",
                                        "cspg_stage2_local_nonrouting_dist_cmp",
                                        "cspg_stage2_routing_pops",
                                        "cspg_stage2_nonrouting_pops",
                                        "cspg_stage2_routing_useless_pops",
                                        "cspg_stage2_nonrouting_useless_pops",
                                        "cspg_stage2_routing_fanout_attempts",
                                        "cspg_stage2_routing_fanout_enqueues",
                                        "cspg_stage2_routing_fanout_skipped_unexpandable",
                                        "cspg_stage2_routing_fanout_skipped_no_unvisited",
                                        "cspg_stage2_routing_fanout_skipped_duplicate_state",
                                        "cspg_stage2_routing_fanout_skipped_frontier_reject",
                                        "cspg_stage2_routing_fanout_skipped_by_bound"});
    if (stats.size() != 21 || stats[0].empty() || stats[1].empty()) {
        return;
    }

    this->total_dist_cmp_.fetch_add(std::stoull(stats[0]), std::memory_order_relaxed);
    this->total_hops_.fetch_add(std::stoull(stats[1]), std::memory_order_relaxed);
    this->total_stat_samples_.fetch_add(1, std::memory_order_relaxed);
    if (!stats[2].empty()) {
        this->total_cspg_phase1_dist_cmp_.fetch_add(std::stoull(stats[2]),
                                                    std::memory_order_relaxed);
    }
    if (!stats[3].empty()) {
        this->total_cspg_phase1_hops_.fetch_add(std::stoull(stats[3]),
                                                std::memory_order_relaxed);
    }
    if (!stats[4].empty()) {
        this->total_cspg_stage2_dist_cmp_.fetch_add(std::stoull(stats[4]),
                                                    std::memory_order_relaxed);
    }
    if (!stats[5].empty()) {
        this->total_cspg_stage2_hops_.fetch_add(std::stoull(stats[5]),
                                                std::memory_order_relaxed);
    }
    if (!stats[6].empty()) {
        this->total_cspg_stage2_local_dist_cmp_.fetch_add(std::stoull(stats[6]),
                                                          std::memory_order_relaxed);
    }
    if (!stats[7].empty()) {
        this->total_cspg_stage2_cross_partition_dist_cmp_.fetch_add(
            std::stoull(stats[7]), std::memory_order_relaxed);
    }
    if (!stats[8].empty()) {
        this->total_cspg_stage2_local_routing_dist_cmp_.fetch_add(
            std::stoull(stats[8]), std::memory_order_relaxed);
    }
    if (!stats[9].empty()) {
        this->total_cspg_stage2_local_nonrouting_dist_cmp_.fetch_add(
            std::stoull(stats[9]), std::memory_order_relaxed);
    }
    if (!stats[10].empty()) {
        this->total_cspg_stage2_routing_pops_.fetch_add(std::stoull(stats[10]),
                                                        std::memory_order_relaxed);
    }
    if (!stats[11].empty()) {
        this->total_cspg_stage2_nonrouting_pops_.fetch_add(std::stoull(stats[11]),
                                                           std::memory_order_relaxed);
    }
    if (!stats[12].empty()) {
        this->total_cspg_stage2_routing_useless_pops_.fetch_add(
            std::stoull(stats[12]), std::memory_order_relaxed);
    }
    if (!stats[13].empty()) {
        this->total_cspg_stage2_nonrouting_useless_pops_.fetch_add(
            std::stoull(stats[13]), std::memory_order_relaxed);
    }
    if (!stats[14].empty()) {
        this->total_cspg_stage2_routing_fanout_attempts_.fetch_add(
            std::stoull(stats[14]), std::memory_order_relaxed);
    }
    if (!stats[15].empty()) {
        this->total_cspg_stage2_routing_fanout_enqueues_.fetch_add(
            std::stoull(stats[15]), std::memory_order_relaxed);
    }
    if (!stats[16].empty()) {
        this->total_cspg_stage2_routing_fanout_skipped_unexpandable_.fetch_add(
            std::stoull(stats[16]), std::memory_order_relaxed);
    }
    if (!stats[17].empty()) {
        this->total_cspg_stage2_routing_fanout_skipped_no_unvisited_.fetch_add(
            std::stoull(stats[17]), std::memory_order_relaxed);
    }
    if (!stats[18].empty()) {
        this->total_cspg_stage2_routing_fanout_skipped_duplicate_state_.fetch_add(
            std::stoull(stats[18]), std::memory_order_relaxed);
    }
    if (!stats[19].empty()) {
        this->total_cspg_stage2_routing_fanout_skipped_frontier_reject_.fetch_add(
            std::stoull(stats[19]), std::memory_order_relaxed);
    }
    if (!stats[20].empty()) {
        this->total_cspg_stage2_routing_fanout_skipped_by_bound_.fetch_add(
            std::stoull(stats[20]), std::memory_order_relaxed);
    }
}

void
SearchEvalCase::do_knn_search() {
    uint64_t topk = config_.top_k;
    auto query_count = this->dataset_ptr_->GetNumberOfQuery();
    this->logger_->Debug("query count is " + std::to_string(query_count));
    auto query_limit = static_cast<uint64_t>(query_count);
    if (config_.search_query_limit > 0) {
        query_limit = std::min(query_limit, config_.search_query_limit);
    }
    auto total_queries =
        config_.search_query_limit > 0 ? query_limit
                                       : std::max(static_cast<uint64_t>(query_count),
                                                  config_.search_query_count);
    for (auto& monitor : this->monitors_) {
        monitor->Start();

        omp_set_num_threads(config_.num_threads_searching);
#pragma omp parallel for schedule(dynamic)
        for (int64_t id = 0; id < static_cast<int64_t>(total_queries); ++id) {
            auto i = id % query_limit;
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(this->dataset_ptr_->GetDim())->Owner(false);
            const void* query_vector = this->dataset_ptr_->GetOneTest(i);
            if (this->dataset_ptr_->GetVectorType() == DENSE_VECTORS) {
                if (this->dataset_ptr_->GetTestDataType() == vsag::DATATYPE_FLOAT32) {
                    query->Float32Vectors((const float*)query_vector);
                } else if (this->dataset_ptr_->GetTestDataType() == vsag::DATATYPE_INT8) {
                    query->Int8Vectors((const int8_t*)query_vector);
                }
            } else {
                query->SparseVectors((const SparseVector*)query_vector);
            }
            auto result = this->index_->KnnSearch(query, topk, config_.search_param);
            if (not result.has_value()) {
                std::cerr << "query error: " << result.error().message << std::endl;
                exit(-1);
            }
            this->collect_search_stats(result.value());
            const int64_t* neighbors = result.value()->GetIds();
            int64_t* ground_truth_neighbors = dataset_ptr_->GetNeighbors(i);
            auto record = std::make_tuple(neighbors,
                                          ground_truth_neighbors,
                                          dataset_ptr_.get(),
                                          query_vector,
                                          result.value()->GetDim());
            monitor->Record(&record);
        }
        monitor->Stop();
    }
}

void
SearchEvalCase::do_range_search() {
}

void
SearchEvalCase::do_knn_filter_search() {
    uint64_t topk = config_.top_k;
    auto query_count = this->dataset_ptr_->GetNumberOfQuery();
    auto train_labels = this->dataset_ptr_->GetTrainLabels();
    auto test_labels = this->dataset_ptr_->GetTestLabels();
    if (train_labels == nullptr) {
        this->logger_->Error("dataset does not contain train_labels");
    }
    if (test_labels == nullptr) {
        this->logger_->Error("dataset does not contain test_labels");
    }
    this->logger_->Debug("query count is " + std::to_string(query_count));
    auto query_limit = query_count;
    if (config_.search_query_limit > 0) {
        query_limit = std::min<int64_t>(query_limit, static_cast<int64_t>(config_.search_query_limit));
    }
    auto total_queries =
        config_.search_query_limit > 0 ? query_limit : std::max<int64_t>(query_count, 10000);
    for (auto& monitor : this->monitors_) {
        monitor->Start();
        for (int64_t id = 0; id < total_queries; ++id) {
            auto i = id % query_limit;
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(this->dataset_ptr_->GetDim())->Owner(false);
            const void* query_vector = this->dataset_ptr_->GetOneTest(i);
            if (this->dataset_ptr_->GetTestDataType() == vsag::DATATYPE_FLOAT32) {
                query->Float32Vectors((const float*)query_vector);
            } else if (this->dataset_ptr_->GetTestDataType() == vsag::DATATYPE_INT8) {
                query->Int8Vectors((const int8_t*)query_vector);
            }
            auto test_label = test_labels[i];
            auto filter = std::make_shared<FilterObj>(
                train_labels, test_label, this->dataset_ptr_->GetValidRatio(test_label));
            auto result = this->index_->KnnSearch(query, topk, config_.search_param, filter);
            if (not result.has_value()) {
                std::cerr << "query error: " << result.error().message << std::endl;
                exit(-1);
            }
            this->collect_search_stats(result.value());
            const int64_t* neighbors = result.value()->GetIds();
            int64_t* ground_truth_neighbors = dataset_ptr_->GetNeighbors(i);
            auto record = std::make_tuple(
                neighbors, ground_truth_neighbors, dataset_ptr_.get(), query_vector, topk);
            monitor->Record(&record);
        }
        monitor->Stop();
    }
}

void
SearchEvalCase::do_range_filter_search() {
}

JsonType
SearchEvalCase::process_result() {
    JsonType result;
    for (auto& monitor : this->monitors_) {
        const auto& one_result = monitor->GetResult();
        EvalCase::MergeJsonType(one_result, result);
    }
    result["action"] = "search";
    result["search_mode"] = config_.search_mode;
    result["index_info"] = JsonType::parse(config_.build_param);
    result["search_param"] = config_.search_param;
    result["index"] = config_.index_name;
    // TODO(deming): remove try-catch after implement GetMemoryUsageDetail
    try {
        result["memory_detail(B)"] = this->index_->GetMemoryUsageDetail();
    } catch (std::runtime_error& e) {
        // if GetMemoryUsageDetail not implemented
        logger_->Error(e.what());
    } catch (vsag::VsagException& e) {
        logger_->Error(e.what());
    }
    const auto stat_samples = this->total_stat_samples_.load(std::memory_order_relaxed);
    if (stat_samples > 0) {
        result["dist_cmp_avg"] =
            static_cast<double>(this->total_dist_cmp_.load(std::memory_order_relaxed)) /
            static_cast<double>(stat_samples);
        result["hops_avg"] =
            static_cast<double>(this->total_hops_.load(std::memory_order_relaxed)) /
            static_cast<double>(stat_samples);
    }
    if (stat_samples > 0 && config_.enable_cspg_stats) {
        result["cspg_phase1_dist_cmp_avg"] =
            static_cast<double>(this->total_cspg_phase1_dist_cmp_.load(std::memory_order_relaxed)) /
            static_cast<double>(stat_samples);
        result["cspg_phase1_hops_avg"] =
            static_cast<double>(this->total_cspg_phase1_hops_.load(std::memory_order_relaxed)) /
            static_cast<double>(stat_samples);
        result["cspg_stage2_dist_cmp_avg"] =
            static_cast<double>(this->total_cspg_stage2_dist_cmp_.load(std::memory_order_relaxed)) /
            static_cast<double>(stat_samples);
        result["cspg_stage2_hops_avg"] =
            static_cast<double>(this->total_cspg_stage2_hops_.load(std::memory_order_relaxed)) /
            static_cast<double>(stat_samples);
        result["cspg_stage2_local_dist_cmp_avg"] = static_cast<double>(
            this->total_cspg_stage2_local_dist_cmp_.load(std::memory_order_relaxed)) /
                                                   static_cast<double>(stat_samples);
        result["cspg_stage2_cross_partition_dist_cmp_avg"] = static_cast<double>(
            this->total_cspg_stage2_cross_partition_dist_cmp_.load(std::memory_order_relaxed)) /
                                                             static_cast<double>(stat_samples);
        result["cspg_stage2_local_routing_dist_cmp_avg"] = static_cast<double>(
            this->total_cspg_stage2_local_routing_dist_cmp_.load(std::memory_order_relaxed)) /
                                                           static_cast<double>(stat_samples);
        result["cspg_stage2_local_nonrouting_dist_cmp_avg"] = static_cast<double>(
            this->total_cspg_stage2_local_nonrouting_dist_cmp_.load(std::memory_order_relaxed)) /
                                                              static_cast<double>(stat_samples);
        result["cspg_stage2_routing_pops_avg"] = static_cast<double>(
            this->total_cspg_stage2_routing_pops_.load(std::memory_order_relaxed)) /
                                                 static_cast<double>(stat_samples);
        result["cspg_stage2_nonrouting_pops_avg"] = static_cast<double>(
            this->total_cspg_stage2_nonrouting_pops_.load(std::memory_order_relaxed)) /
                                                    static_cast<double>(stat_samples);
        result["cspg_stage2_routing_useless_pops_avg"] = static_cast<double>(
            this->total_cspg_stage2_routing_useless_pops_.load(std::memory_order_relaxed)) /
                                                          static_cast<double>(stat_samples);
        result["cspg_stage2_nonrouting_useless_pops_avg"] = static_cast<double>(
            this->total_cspg_stage2_nonrouting_useless_pops_.load(std::memory_order_relaxed)) /
                                                             static_cast<double>(stat_samples);
        result["cspg_stage2_routing_fanout_attempts_avg"] = static_cast<double>(
            this->total_cspg_stage2_routing_fanout_attempts_.load(std::memory_order_relaxed)) /
                                                             static_cast<double>(stat_samples);
        result["cspg_stage2_routing_fanout_enqueues_avg"] = static_cast<double>(
            this->total_cspg_stage2_routing_fanout_enqueues_.load(std::memory_order_relaxed)) /
                                                             static_cast<double>(stat_samples);
        result["cspg_stage2_routing_fanout_skipped_unexpandable_avg"] =
            static_cast<double>(
                this->total_cspg_stage2_routing_fanout_skipped_unexpandable_.load(
                    std::memory_order_relaxed)) /
            static_cast<double>(stat_samples);
        result["cspg_stage2_routing_fanout_skipped_no_unvisited_avg"] = static_cast<double>(
            this->total_cspg_stage2_routing_fanout_skipped_no_unvisited_.load(
                std::memory_order_relaxed)) /
                                                                      static_cast<double>(
                                                                          stat_samples);
        result["cspg_stage2_routing_fanout_skipped_duplicate_state_avg"] =
            static_cast<double>(
                this->total_cspg_stage2_routing_fanout_skipped_duplicate_state_.load(
                    std::memory_order_relaxed)) /
            static_cast<double>(stat_samples);
        result["cspg_stage2_routing_fanout_skipped_frontier_reject_avg"] =
            static_cast<double>(
                this->total_cspg_stage2_routing_fanout_skipped_frontier_reject_.load(
                    std::memory_order_relaxed)) /
            static_cast<double>(stat_samples);
        result["cspg_stage2_routing_fanout_skipped_by_bound_avg"] = static_cast<double>(
            this->total_cspg_stage2_routing_fanout_skipped_by_bound_.load(
                std::memory_order_relaxed)) /
                                                                     static_cast<double>(
                                                                         stat_samples);
    }
    EvalCase::MergeJsonType(this->basic_info_, result);
    if (config_.build_base_limit > 0) {
        auto base_count = this->dataset_ptr_->GetNumberOfBase();
        result["dataset_info"]["base_count"] =
            std::min<uint64_t>(static_cast<uint64_t>(base_count), config_.build_base_limit);
    }
    return result;
}

}  // namespace vsag::eval
