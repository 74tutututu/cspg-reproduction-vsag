
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

#include "hgraph.h"

#include <datacell/compressed_graph_datacell_parameter.h>
#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <queue>
#include <stdexcept>

#include "algorithm/inner_index_interface.h"
#include "analyzer/analyzer.h"
#include "attr/argparse.h"
#include "common.h"
#include "datacell/flatten_interface.h"
#include "datacell/sparse_graph_datacell.h"
#include "dataset_impl.h"
#include "hgraph_shrink_context.h"
#include "impl/filter/filter_headers.h"
#include "impl/heap/standard_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "index/index_impl.h"
#include "index/iterator_filter.h"
#include "io/reader_io_parameter.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"
#include "typing.h"
#include "utils/util_functions.h"
#include "utils/visited_list.h"
#include "vsag/options.h"

namespace vsag {
namespace {

constexpr int kCspgRoutingPartition = -1;
constexpr int kCspgUnassignedPartition = -2;
constexpr uint32_t kCspgVisitPrefetchStride = 3;

GraphInterfaceParamPtr
clone_graph_param_with_max_degree(const GraphInterfaceParamPtr& source, uint64_t max_degree) {
    if (source == nullptr) {
        return nullptr;
    }
    auto json = source->ToJson();
    json[HGRAPH_GRAPH_MAX_DEGREE].SetInt(static_cast<int64_t>(max_degree));
    return GraphInterfaceParameter::GetGraphParameterByJson(source->graph_storage_type_, json);
}

GraphInterfaceParamPtr
create_cspg_partition_graph_param(const GraphInterfaceParamPtr& source,
                                  uint64_t max_degree,
                                  bool use_sparse_storage) {
    if (source == nullptr) {
        return nullptr;
    }
    if (!use_sparse_storage) {
        return clone_graph_param_with_max_degree(source, max_degree);
    }

    auto sparse_param = std::make_shared<SparseGraphDatacellParameter>();
    sparse_param->max_degree_ = max_degree;
    if (source->graph_storage_type_ == GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT) {
        auto graph_param = std::dynamic_pointer_cast<GraphDataCellParameter>(source);
        if (graph_param != nullptr) {
            sparse_param->remove_flag_bit_ = graph_param->remove_flag_bit_;
            sparse_param->support_delete_ = graph_param->support_remove_;
        }
    }
    return sparse_param;
}

SparseGraphDatacellParamPtr
clone_sparse_graph_param_with_max_degree(const SparseGraphDatacellParamPtr& source,
                                         uint64_t max_degree) {
    auto cloned = std::make_shared<SparseGraphDatacellParameter>();
    if (source != nullptr) {
        cloned->support_delete_ = source->support_delete_;
        cloned->remove_flag_bit_ = source->remove_flag_bit_;
    }
    cloned->max_degree_ = max_degree;
    return cloned;
}

// CSPG 阶段 1 分区过滤器：仅允许同分区或路由点通过。
class CspgPartitionFilter : public Filter {
public:
    CspgPartitionFilter(const std::vector<int>* partitions, int partition_id)
        : partitions_(partitions), partition_id_(partition_id) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        if (partitions_ == nullptr || id < 0 || static_cast<size_t>(id) >= partitions_->size()) {
            return false;
        }
        int partition = (*partitions_)[static_cast<size_t>(id)];
        // 路由点(-1)或同分区点允许通过。
        return partition == kCspgRoutingPartition || partition == partition_id_;
    }

private:
    const std::vector<int>* partitions_{nullptr};
    int partition_id_{0};
};

std::vector<InnerIdType>
BuildCspgPartitionEntryPoints(const std::vector<int>& partitions, int partition_count) {
    std::vector<InnerIdType> entry_points(std::max(1, partition_count), INVALID_ENTRY_POINT);
    InnerIdType routing_entry = INVALID_ENTRY_POINT;

    for (size_t id = 0; id < partitions.size(); ++id) {
        const int partition = partitions[id];
        if (partition == kCspgRoutingPartition) {
            if (routing_entry == INVALID_ENTRY_POINT) {
                routing_entry = static_cast<InnerIdType>(id);
            }
            continue;
        }

        if (partition >= 0 && static_cast<size_t>(partition) < entry_points.size() &&
            entry_points[partition] == INVALID_ENTRY_POINT) {
            entry_points[partition] = static_cast<InnerIdType>(id);
        }
    }

    if (routing_entry != INVALID_ENTRY_POINT) {
        for (auto& entry_point : entry_points) {
            if (entry_point == INVALID_ENTRY_POINT) {
                entry_point = routing_entry;
            }
        }
    }

    return entry_points;
}

}  // namespace
}  // namespace vsag

namespace vsag {

class HGraphAnalyzer;

HGraph::HGraph(const HGraphParameterPtr& hgraph_param, const vsag::IndexCommonParam& common_param)
    : InnerIndexInterface(hgraph_param, common_param),
      route_graphs_(common_param.allocator_.get()),
      cspg_partition_graphs_(common_param.allocator_.get()),
      use_elp_optimizer_(hgraph_param->use_elp_optimizer),
      ignore_reorder_(hgraph_param->ignore_reorder),
      build_by_base_(hgraph_param->build_by_base),
      ef_construct_(hgraph_param->ef_construction),
      alpha_(hgraph_param->alpha),
      odescent_param_(hgraph_param->odescent_param),
      graph_type_(hgraph_param->graph_type),
      bottom_graph_param_(hgraph_param->bottom_graph_param),
      hierarchical_datacell_param_(hgraph_param->hierarchical_graph_param),
      use_old_serial_format_(common_param.use_old_serial_format_) {
    this->label_table_->compress_duplicate_data_ = hgraph_param->support_duplicate;
    this->label_table_->support_tombstone_ = hgraph_param->support_tombstone;
    neighbors_mutex_ = std::make_shared<PointsMutex>(0, common_param.allocator_.get());
    this->basic_flatten_codes_ =
        FlattenInterface::MakeInstance(hgraph_param->base_codes_param, common_param);
    if (use_reorder_) {
        this->high_precise_codes_ =
            FlattenInterface::MakeInstance(hgraph_param->precise_codes_param, common_param);
    }
    this->searcher_ = std::make_shared<BasicSearcher>(common_param, neighbors_mutex_);

    this->bottom_graph_ =
        GraphInterface::MakeInstance(hgraph_param->bottom_graph_param, common_param);
    mult_ = 1 / log(1.0 * static_cast<double>(this->bottom_graph_->MaximumDegree()));

    init_resize_bit_and_reorder();

    this->parallel_searcher_ =
        std::make_shared<ParallelSearcher>(common_param, thread_pool_, neighbors_mutex_);

    UnorderedMap<std::string, float> default_param(common_param.allocator_.get());
    default_param.insert(
        {PREFETCH_DEPTH_CODE, (this->basic_flatten_codes_->code_size_ + 63.0) / 64.0});
    this->basic_flatten_codes_->SetRuntimeParameters(default_param);

    if (use_elp_optimizer_) {
        optimizer_ = std::make_shared<Optimizer<BasicSearcher>>(common_param);
    }
    check_and_init_raw_vector(hgraph_param->raw_vector_param, common_param);
    resize(bottom_graph_->max_capacity_);

    // 从参数对象读取 CSPG 参数，影响分区与路由点比例。
    this->cspg_m_ = hgraph_param->cspg_m;
    this->cspg_lambda_ = hgraph_param->cspg_lambda;
    this->cspg_partition_graph_type_ = hgraph_param->cspg_partition_graph_type;
    const auto cspg_partition_max_degree =
        static_cast<uint64_t>(hgraph_param->ResolveCspgPartitionMaxDegree());
    this->cspg_partition_graph_param_ = create_cspg_partition_graph_param(
        this->bottom_graph_param_, cspg_partition_max_degree, true);
    this->cspg_partition_hierarchical_datacell_param_ = clone_sparse_graph_param_with_max_degree(
        this->hierarchical_datacell_param_, std::max<uint64_t>(1, cspg_partition_max_degree / 2));
    this->cspg_partition_entry_points_.assign(std::max(1, this->cspg_m_), INVALID_ENTRY_POINT);
    this->cspg_partition_route_entry_points_.assign(std::max(1, this->cspg_m_),
                                                    INVALID_ENTRY_POINT);
    this->cspg_partition_route_entry_levels_.assign(std::max(1, this->cspg_m_), -1);
    this->staged_node_partition_.assign(this->max_capacity_.load(), kCspgUnassignedPartition);

    this->common_param_.metric_ = common_param.metric_;
    this->common_param_.data_type_ = common_param.data_type_;
    this->common_param_.dim_ = common_param.dim_;
    this->common_param_.extra_info_size_ = common_param.extra_info_size_;
    this->common_param_.allocator_ = common_param.allocator_;
    this->common_param_.thread_pool_ = common_param.thread_pool_;
    this->common_param_.use_old_serial_format_ = common_param.use_old_serial_format_;

    this->ensure_cspg_partition_graphs();
    this->ensure_cspg_partition_route_graphs();
}
void
HGraph::Train(const DatasetPtr& base) {
    int64_t total_elements = base->GetNumElements();
    int64_t dim = base->GetDim();
    DatasetPtr train_data =
        vsag::sample_train_data(base, total_elements, dim, train_sample_count_, allocator_);

    const auto* data_ptr = get_data(train_data);
    this->basic_flatten_codes_->Train(data_ptr, train_data->GetNumElements());
    if (use_reorder_) {
        this->high_precise_codes_->Train(data_ptr, train_data->GetNumElements());
    }
    if (create_new_raw_vector_) {
        // nothing to do since raw_vector_ is fp32
        this->raw_vector_->Train(data_ptr, train_data->GetNumElements());
    }
}

std::vector<int64_t>
HGraph::Build(const DatasetPtr& data) {
    CHECK_ARGUMENT(GetNumElements() == 0, "index is not empty");
    this->Train(data);
    std::vector<int64_t> ret;
    if (this->is_cspg_enabled()) {
        ret = this->build_cspg_by_partition(data);
    } else if (graph_type_ == GRAPH_TYPE_VALUE_NSW) {
        ret = this->Add(data);
    } else {
        ret = this->build_by_odescent(data);
    }
    if (use_elp_optimizer_) {
        elp_optimize();
    }
    return ret;
}

JsonType
HGraph::map_hgraph_param(const JsonType& hgraph_json) {
    static const ConstParamMap external_mapping = {
        {"cspg_m", {"cspg_m"}},

        {"cspg_lambda", {"cspg_lambda"}},

        {HGRAPH_CSPG_PARTITION_MAX_DEGREE, {HGRAPH_CSPG_PARTITION_MAX_DEGREE}},

        {HGRAPH_CSPG_PARTITION_GRAPH_TYPE, {HGRAPH_CSPG_PARTITION_GRAPH_TYPE}},

        {
            HGRAPH_USE_REORDER,
            {
                USE_REORDER_KEY,
            },
        },
        {
            HGRAPH_USE_ELP_OPTIMIZER,
            {
                HGRAPH_USE_ELP_OPTIMIZER_KEY,
            },
        },
        {
            HGRAPH_IGNORE_REORDER,
            {
                HGRAPH_IGNORE_REORDER_KEY,
            },
        },
        {
            HGRAPH_BUILD_BY_BASE_QUANTIZATION,
            {
                HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY,
            },
        },
        {
            USE_ATTRIBUTE_FILTER,
            {
                USE_ATTRIBUTE_FILTER_KEY,
            },
        },
        {
            HGRAPH_BASE_QUANTIZATION_TYPE,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            STORE_RAW_VECTOR,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                HOLD_MOLDS,
            },
        },
        {
            HGRAPH_BASE_IO_TYPE,
            {
                BASE_CODES_KEY,
                IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            HGRAPH_PRECISE_IO_TYPE,
            {
                PRECISE_CODES_KEY,
                IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            HGRAPH_BASE_FILE_PATH,
            {
                BASE_CODES_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            HGRAPH_PRECISE_FILE_PATH,
            {
                PRECISE_CODES_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            HGRAPH_PRECISE_QUANTIZATION_TYPE,
            {
                PRECISE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            HGRAPH_GRAPH_IO_TYPE,
            {
                GRAPH_KEY,
                IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            HGRAPH_GRAPH_FILE_PATH,
            {
                GRAPH_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            STORE_RAW_VECTOR,
            {
                PRECISE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                HOLD_MOLDS,
            },
        },
        {
            STORE_RAW_VECTOR,
            {
                STORE_RAW_VECTOR_KEY,
            },
        },
        {
            RAW_VECTOR_IO_TYPE,
            {
                RAW_VECTOR_KEY,
                IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            RAW_VECTOR_FILE_PATH,
            {
                RAW_VECTOR_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            HGRAPH_GRAPH_MAX_DEGREE,
            {
                GRAPH_KEY,
                GRAPH_PARAM_MAX_DEGREE_KEY,
            },
        },
        {
            HGRAPH_BUILD_EF_CONSTRUCTION,
            {
                EF_CONSTRUCTION_KEY,
            },
        },
        {
            HGRAPH_BUILD_ALPHA,
            {
                ALPHA_KEY,
            },
        },
        {
            HGRAPH_INIT_CAPACITY,
            {
                GRAPH_KEY,
                GRAPH_PARAM_INIT_MAX_CAPACITY_KEY,
            },
        },
        {
            HGRAPH_GRAPH_TYPE,
            {
                GRAPH_KEY,
                GRAPH_TYPE_KEY,
            },
        },
        {
            HGRAPH_GRAPH_STORAGE_TYPE,
            {
                GRAPH_KEY,
                GRAPH_STORAGE_TYPE_KEY,
            },
        },
        {
            ODESCENT_PARAMETER_ALPHA,
            {
                GRAPH_KEY,
                ODESCENT_PARAMETER_ALPHA,
            },
        },
        {
            ODESCENT_PARAMETER_GRAPH_ITER_TURN,
            {
                GRAPH_KEY,
                ODESCENT_PARAMETER_GRAPH_ITER_TURN,
            },
        },
        {
            ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE,
            {
                GRAPH_KEY,
                ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE,
            },
        },
        {
            ODESCENT_PARAMETER_MIN_IN_DEGREE,
            {
                GRAPH_KEY,
                ODESCENT_PARAMETER_MIN_IN_DEGREE,
            },
        },
        {
            ODESCENT_PARAMETER_BUILD_BLOCK_SIZE,
            {
                GRAPH_KEY,
                ODESCENT_PARAMETER_BUILD_BLOCK_SIZE,
            },
        },
        {
            HGRAPH_BUILD_THREAD_COUNT,
            {
                BUILD_THREAD_COUNT_KEY,
            },
        },
        {
            SQ4_UNIFORM_TRUNC_RATE,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY,
            },
        },
        {
            RABITQ_PCA_DIM,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                PCA_DIM_KEY,
            },
        },
        {
            INDEX_TQ_CHAIN,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                TQ_CHAIN_KEY,
            },
        },
        {
            RABITQ_BITS_PER_DIM_QUERY,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY,
            },
        },
        {
            RABITQ_BITS_PER_DIM_BASE,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY,
            },
        },
        {
            HGRAPH_BASE_PQ_DIM,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                PRODUCT_QUANTIZATION_DIM_KEY,
            },
        },
        {
            RABITQ_USE_FHT,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                USE_FHT_KEY,
            },
        },
        {
            HGRAPH_SUPPORT_REMOVE,
            {GRAPH_KEY, GRAPH_SUPPORT_REMOVE},
        },
        {
            HGRAPH_REMOVE_FLAG_BIT,
            {GRAPH_KEY, REMOVE_FLAG_BIT},
        },
        {
            HGRAPH_SUPPORT_DUPLICATE,
            {
                SUPPORT_DUPLICATE,
            },
        },
        {
            HGRAPH_SUPPORT_TOMBSTONE,
            {
                SUPPORT_TOMBSTONE,
            },
        }};
    const std::string hgraph_params_template =
        R"(
    {
        "{TYPE_KEY}": "{INDEX_TYPE_HGRAPH}",
        "{USE_REORDER_KEY}": false,
        "{HGRAPH_USE_ENV_OPTIMIZER}": false,
        "{HGRAPH_IGNORE_REORDER_KEY}": false,
        "{HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY}": false,
        "{HGRAPH_USE_ATTRIBUTE_FILTER_KEY}": false,
        "{GRAPH_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{GRAPH_TYPE_KEY}": "{GRAPH_TYPE_VALUE_NSW}",
            "{GRAPH_STORAGE_TYPE_KEY}": "{GRAPH_STORAGE_TYPE_VALUE_FLAT}",
            "{ODESCENT_PARAMETER_BUILD_BLOCK_SIZE}": 10000,
            "{ODESCENT_PARAMETER_MIN_IN_DEGREE}": 1,
            "{ODESCENT_PARAMETER_ALPHA}": 1.2,
            "{ODESCENT_PARAMETER_GRAPH_ITER_TURN}": 30,
            "{ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE}": 0.2,
            "{GRAPH_PARAM_MAX_DEGREE_KEY}": 64,
            "{GRAPH_PARAM_INIT_MAX_CAPACITY_KEY}": 100,
            "{GRAPH_SUPPORT_REMOVE}": false,
            "{REMOVE_FLAG_BIT}": 8
        },
        "{BASE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "flatten",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY}": 0.05,
                "{PCA_DIM_KEY}": 0,
                "{RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY}": 32,
                "{TQ_CHAIN_KEY}": "",
                "nbits": 8,
                "{PRODUCT_QUANTIZATION_DIM_KEY}": 1,
                "{HOLD_MOLDS}": false
            }
        },
        "{PRECISE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "flatten",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY}": 0.05,
                "{PCA_DIM_KEY}": 0,
                "{PRODUCT_QUANTIZATION_DIM_KEY}": 1,
                "{HOLD_MOLDS}": false
            }
        },
        "{STORE_RAW_VECTOR_KEY}": false,
        "{RAW_VECTOR_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "flatten",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{HOLD_MOLDS}": true
            }
        },
        "{BUILD_THREAD_COUNT_KEY}": 100,
        "{EXTRA_INFO_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            }
        },
        "{ATTR_PARAMS_KEY}": {
            "{ATTR_HAS_BUCKETS_KEY}": false
        },
        "{HGRAPH_SUPPORT_DUPLICATE}": false,
        "{HGRAPH_SUPPORT_TOMBSTONE}": false,
        "{EF_CONSTRUCTION_KEY}": 400
    })";

    std::string str = format_map(hgraph_params_template, DEFAULT_MAP);
    auto inner_json = JsonType::Parse(str);
    mapping_external_param_to_inner(hgraph_json, external_mapping, inner_json);

    return inner_json;
}

bool
HGraph::Tune(const std::string& parameters, bool disable_future_tuning) {
    if (not this->index_feature_list_->CheckFeature(IndexFeature::SUPPORT_TUNE) or
        not this->has_raw_vector_) {
        return false;
    }

    // parse
    auto parsed_params = JsonType::Parse(parameters);
    JsonType hgraph_json;
    if (parsed_params.Contains(INDEX_PARAM)) {
        hgraph_json = parsed_params[INDEX_PARAM];
    }

    // map
    auto inner_json = map_hgraph_param(hgraph_json);

    // construct param obj
    auto hgraph_parameter = std::make_shared<HGraphParameter>();
    hgraph_parameter->FromJson(inner_json);
    auto inner_parameter = std::make_shared<InnerIndexParameter>();
    inner_parameter->FromJson(inner_json);

    // init new_basic_code obj
    auto common_param = this->basic_flatten_codes_->ExportCommonParam();
    auto new_basic_code =
        FlattenInterface::MakeInstance(hgraph_parameter->base_codes_param, common_param);
    FlattenInterfacePtr new_precise_code;
    if (inner_parameter->use_reorder) {
        new_precise_code =
            FlattenInterface::MakeInstance(hgraph_parameter->precise_codes_param, common_param);
    }

    std::scoped_lock lock(this->add_mutex_);

    // check which code need to tune and update create_param_ptr_
    bool is_tune_base_code = false;
    bool is_tune_precise_code = false;
    auto param = std::dynamic_pointer_cast<HGraphParameter>(create_param_ptr_);
    if (basic_flatten_codes_->GetQuantizerName() != new_basic_code->GetQuantizerName()) {
        // [case 1] base_code is not same
        is_tune_base_code = true;
    }
    if (use_reorder_ and inner_parameter->use_reorder and
        this->high_precise_codes_->GetQuantizerName() != new_precise_code->GetQuantizerName()) {
        // [case 2] precise code is not same
        is_tune_precise_code = true;
    }
    if (not inner_parameter->use_reorder) {
        // [case 3] drop precise_code
        use_reorder_ = false;
        this->high_precise_codes_.reset();
        param->precise_codes_param.reset();
        is_tune_precise_code = false;
    }
    if (not use_reorder_ and inner_parameter->use_reorder) {
        // [case 4] assign new precise_code
        use_reorder_ = true;
        is_tune_precise_code = true;
    }

    // update create_param_ptr_
    if (is_tune_base_code) {
        param->base_codes_param = hgraph_parameter->base_codes_param;
    }
    if (is_tune_precise_code) {
        param->precise_codes_param = hgraph_parameter->precise_codes_param;
    }
    param->use_reorder = use_reorder_;

    // export train data and train new_basic_code
    auto train_count = std::min(this->train_sample_count_, this->GetNumElements());
    Vector<float> train_data(train_count * dim_, 0, allocator_);
    if (is_tune_base_code or is_tune_precise_code) {
        for (InnerIdType i = 0; i < train_count; i++) {
            this->GetVectorByInnerId(i, (train_data.data() + i * dim_));
        }
    }

    auto tune_and_rebuild =
        [&](bool need_tune, FlattenInterfacePtr old_code, FlattenInterfacePtr new_code) {
            if (not need_tune) {
                return old_code;
            }

            new_code->Train(train_data.data(), train_count);

            Vector<float> insert_buffer(dim_, 0, allocator_);
            for (int64_t i = 0; i < total_count_; ++i) {
                GetVectorByInnerId(i, insert_buffer.data());
                new_code->InsertVector(static_cast<const void*>(insert_buffer.data()), i);
            }
            return new_code;
        };

    basic_flatten_codes_ =
        tune_and_rebuild(is_tune_base_code, basic_flatten_codes_, new_basic_code);
    high_precise_codes_ =
        tune_and_rebuild(is_tune_precise_code, high_precise_codes_, new_precise_code);

    check_and_init_raw_vector(param->raw_vector_param, common_param, false);
    init_resize_bit_and_reorder();

    // set status
    if (disable_future_tuning) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_TUNE, false);
        this->raw_vector_.reset();
        has_raw_vector_ = false;
        create_new_raw_vector_ = false;
    }
    return true;
}

std::vector<int64_t>
HGraph::build_by_odescent(const DatasetPtr& data) {
    std::vector<int64_t> failed_ids;

    auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* vectors = data->GetFloat32Vectors();
    const auto* extra_infos = data->GetExtraInfos();
    auto inner_ids = this->get_unique_inner_ids(total);
    Vector<Vector<InnerIdType>> route_graph_ids(allocator_);
    InnerIdType cur_size = 0;
    for (int64_t i = 0; i < total; ++i) {
        auto label = labels[i];
        if (this->label_table_->CheckLabel(label)) {
            failed_ids.emplace_back(label);
            continue;
        }
        InnerIdType inner_id = inner_ids.at(cur_size);
        cur_size++;
        this->label_table_->Insert(inner_id, label);
        this->basic_flatten_codes_->InsertVector(vectors + dim_ * i, inner_id);
        if (use_reorder_) {
            this->high_precise_codes_->InsertVector(vectors + dim_ * i, inner_id);
        }
        if (create_new_raw_vector_) {
            this->raw_vector_->InsertVector(vectors + dim_ * i, inner_id);
        }
        auto level = this->get_random_level() - 1;
        if (level >= 0) {
            if (level >= static_cast<int>(route_graph_ids.size()) || route_graph_ids.empty()) {
                for (auto k = static_cast<int>(route_graph_ids.size()); k <= level; ++k) {
                    route_graph_ids.emplace_back(allocator_);
                }
                entry_point_id_ = inner_id;
            }
            for (int j = 0; j <= level; ++j) {
                route_graph_ids[j].emplace_back(inner_id);
            }
        }
    }
    this->resize(total_count_);
    auto build_data = (use_reorder_ and not build_by_base_) ? this->high_precise_codes_
                                                            : this->basic_flatten_codes_;
    {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree();
        ODescent odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        odescent_builder.Build();
        odescent_builder.SaveGraph(bottom_graph_);
    }
    for (auto& route_graph_id : route_graph_ids) {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree() / 2;
        ODescent sparse_odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        auto graph = this->generate_one_route_graph();
        sparse_odescent_builder.Build(route_graph_id);
        sparse_odescent_builder.SaveGraph(graph);
        this->route_graphs_.emplace_back(graph);
    }
    return failed_ids;
}

std::vector<int64_t>
HGraph::build_cspg_by_partition(const DatasetPtr& data) {
    std::vector<int64_t> failed_ids;

    const auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* extra_infos = data->GetExtraInfos();
    const auto* attr_sets = data->GetAttributeSets();
    auto inner_ids = this->get_unique_inner_ids(total);

    struct PendingNode {
        InnerIdType inner_id;
        int64_t data_index;
    };

    std::vector<PendingNode> pending_nodes;
    pending_nodes.reserve(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i) {
        const auto label = labels[i];
        if (this->label_table_->CheckLabel(label)) {
            failed_ids.emplace_back(label);
            continue;
        }

        InnerIdType inner_id = inner_ids.at(pending_nodes.size());
        this->label_table_->Insert(inner_id, label);
        const auto* vector_data = this->get_data(data, i);
        this->basic_flatten_codes_->InsertVector(vector_data, inner_id);
        if (use_reorder_) {
            this->high_precise_codes_->InsertVector(vector_data, inner_id);
        }
        if (create_new_raw_vector_) {
            this->raw_vector_->InsertVector(vector_data, inner_id);
        }
        if (this->extra_infos_ != nullptr && extra_infos != nullptr) {
            this->extra_infos_->InsertExtraInfo(extra_infos + i * extra_info_size_, inner_id);
        }
        if (attr_sets != nullptr && this->use_attribute_filter_) {
            this->attr_filter_index_->Insert(attr_sets[i], inner_id);
        }

        pending_nodes.push_back({inner_id, i});
    }

    this->resize(total_count_);
    if (pending_nodes.empty()) {
        return failed_ids;
    }
    if (this->odescent_param_ == nullptr) {
        this->odescent_param_ = std::make_shared<ODescentParameter>();
    }

    const size_t partition_count = static_cast<size_t>(std::max(1, this->cspg_m_));
    this->node_partition_.assign(this->max_capacity_.load(), kCspgUnassignedPartition);
    this->staged_node_partition_.assign(this->max_capacity_.load(), kCspgUnassignedPartition);
    this->route_graphs_.clear();
    this->cspg_partition_graphs_.clear();
    this->ensure_cspg_partition_graphs();
    this->cspg_partition_route_graphs_.assign(partition_count, {});
    this->cspg_partition_route_entry_points_.assign(partition_count, INVALID_ENTRY_POINT);
    this->cspg_partition_route_entry_levels_.assign(partition_count, -1);

    std::vector<size_t> shuffled_order(pending_nodes.size());
    for (size_t i = 0; i < shuffled_order.size(); ++i) {
        shuffled_order[i] = i;
    }
    std::mt19937 generator(std::random_device{}());
    std::shuffle(shuffled_order.begin(), shuffled_order.end(), generator);

    const size_t routing_count = static_cast<size_t>(std::floor(
        static_cast<double>(pending_nodes.size()) * static_cast<double>(this->cspg_lambda_)));
    std::uniform_int_distribution<int> partition_dist(0, std::max(1, this->cspg_m_) - 1);

    std::vector<InnerIdType> routing_ids;
    routing_ids.reserve(routing_count);
    std::vector<std::vector<InnerIdType>> partition_local_ids(partition_count);
    std::vector<int> route_levels(this->max_capacity_.load(), -1);
    for (size_t rank = 0; rank < shuffled_order.size(); ++rank) {
        const auto inner_id = pending_nodes[shuffled_order[rank]].inner_id;
        route_levels[inner_id] = this->get_random_level() - 1;
        if (rank < routing_count) {
            this->node_partition_[inner_id] = kCspgRoutingPartition;
            routing_ids.push_back(inner_id);
            continue;
        }
        const int partition_id = partition_dist(generator);
        this->node_partition_[inner_id] = partition_id;
        partition_local_ids[static_cast<size_t>(partition_id)].push_back(inner_id);
    }
    auto build_data = (use_reorder_ && !build_by_base_) ? this->high_precise_codes_
                                                        : this->basic_flatten_codes_;
    const auto original_max_degree = this->odescent_param_->max_degree;
    this->cspg_partition_entry_points_.assign(partition_count, INVALID_ENTRY_POINT);
    const bool build_partition_graphs_by_nsw =
        this->cspg_partition_graph_type_ == GRAPH_TYPE_VALUE_NSW;
    for (auto& partition_graph : this->cspg_partition_graphs_) {
        if (partition_graph != nullptr) {
            partition_graph->Resize(this->max_capacity_.load());
        }
    }
    if (build_partition_graphs_by_nsw) {
        for (size_t rank = 0; rank < shuffled_order.size(); ++rank) {
            const auto& node = pending_nodes[shuffled_order[rank]];
            const auto* vector_data = this->get_data(data, node.data_index);
            const auto level =
                node.inner_id < route_levels.size() ? route_levels[node.inner_id] : -1;
            this->graph_add_one(vector_data, level, node.inner_id);
        }
    }
    for (size_t partition_id = 0; partition_id < partition_count; ++partition_id) {
        Vector<InnerIdType> member_ids(this->allocator_);
        member_ids.reserve(routing_ids.size() + partition_local_ids[partition_id].size());
        for (auto inner_id : routing_ids) {
            member_ids.push_back(inner_id);
        }
        for (auto inner_id : partition_local_ids[partition_id]) {
            member_ids.push_back(inner_id);
        }

        auto partition_graph = this->cspg_partition_graphs_[partition_id];
        partition_graph->Resize(this->max_capacity_.load());
        if (member_ids.empty()) {
            continue;
        }

        if (!build_partition_graphs_by_nsw) {
            std::uniform_int_distribution<size_t> partition_entry_dist(0, member_ids.size() - 1);
            this->cspg_partition_entry_points_[partition_id] =
                member_ids[partition_entry_dist(generator)];

            this->odescent_param_->max_degree = partition_graph->MaximumDegree();
            ODescent odescent_builder(
                this->odescent_param_, build_data, allocator_, this->thread_pool_.get());
            odescent_builder.Build(member_ids);
            odescent_builder.SaveGraph(partition_graph);
        }

        Vector<Vector<InnerIdType>> route_graph_ids(allocator_);
        for (const auto inner_id : member_ids) {
            const auto level = inner_id < route_levels.size() ? route_levels[inner_id] : -1;
            if (level < 0) {
                continue;
            }
            while (route_graph_ids.size() <= static_cast<size_t>(level)) {
                route_graph_ids.emplace_back(allocator_);
            }
            for (int route_level = 0; route_level <= level; ++route_level) {
                route_graph_ids[route_level].emplace_back(inner_id);
            }
        }
        if (!route_graph_ids.empty()) {
            this->ensure_cspg_partition_route_levels(static_cast<int>(partition_id),
                                                     route_graph_ids.size());
            this->odescent_param_->max_degree = std::max<uint64_t>(
                1, this->cspg_partition_hierarchical_datacell_param_->max_degree_);
            for (size_t route_level = 0; route_level < route_graph_ids.size(); ++route_level) {
                auto route_graph =
                    this->cspg_partition_route_graphs_[partition_id][route_level];
                if (route_graph == nullptr || route_graph_ids[route_level].empty()) {
                    continue;
                }
                ODescent route_builder(
                    this->odescent_param_, build_data, allocator_, this->thread_pool_.get());
                route_builder.Build(route_graph_ids[route_level]);
                route_builder.SaveGraph(route_graph);
            }
        }
    }
    this->odescent_param_->max_degree = original_max_degree;
    this->rebuild_cspg_partition_route_entries();

    auto fallback_partition_entry_points =
        BuildCspgPartitionEntryPoints(this->node_partition_, this->cspg_m_);
    if (this->cspg_partition_entry_points_.size() < fallback_partition_entry_points.size()) {
        this->cspg_partition_entry_points_.resize(fallback_partition_entry_points.size(),
                                                  INVALID_ENTRY_POINT);
    }
    for (size_t partition_id = 0; partition_id < fallback_partition_entry_points.size();
         ++partition_id) {
        if (this->cspg_partition_entry_points_[partition_id] == INVALID_ENTRY_POINT) {
            this->cspg_partition_entry_points_[partition_id] =
                fallback_partition_entry_points[partition_id];
        }
    }
    this->entry_point_id_ = this->choose_any_cspg_entry_point();
    return failed_ids;
}

std::vector<int64_t>
HGraph::Add(const DatasetPtr& data, AddMode mode) {
    std::vector<int64_t> failed_ids;
    bool is_initial_build = (this->GetNumElements() == 0);
    auto base_dim = data->GetDim();
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(base_dim == dim_,
                       fmt::format("base.dim({}) must be equal to index.dim({})", base_dim, dim_));
    }
    CHECK_ARGUMENT(get_data(data) != nullptr, "base.float_vector is nullptr");

    {
        std::scoped_lock lock(this->add_mutex_);
        if (this->total_count_ == 0) {
            this->Train(data);
        }
    }

    auto add_func = [&](const void* data,
                        int level,
                        InnerIdType inner_id,
                        const char* extra_info,
                        const AttributeSet* attrs) -> void {
        if (this->extra_infos_ != nullptr) {
            this->extra_infos_->InsertExtraInfo(extra_info, inner_id);
        }
        if (attrs != nullptr and this->use_attribute_filter_) {
            this->attr_filter_index_->Insert(*attrs, inner_id);
        }
        this->add_one_point(data, level, inner_id);
    };

    std::vector<std::future<void>> futures;
    auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* extra_infos = data->GetExtraInfos();
    const auto* attr_sets = data->GetAttributeSets();
    Vector<std::pair<InnerIdType, LabelType>> inner_ids(allocator_);
    for (int64_t j = 0; j < total; ++j) {
        InnerIdType inner_id;

        // try recover tombstone
        if (this->data_type_ != DataTypes::DATA_TYPE_SPARSE) {
            auto one_base = get_single_dataset(data, j);
            bool is_process_finished = try_recover_tombstone(one_base, failed_ids);
            if (is_process_finished) {
                continue;
            }
        }

        {
            std::scoped_lock lock(this->add_mutex_);
            inner_id = this->get_unique_inner_ids(1).at(0);
            uint64_t new_count = total_count_;
            this->resize(new_count);
        }

        {
            std::scoped_lock label_lock(this->label_lookup_mutex_);
            this->label_table_->Insert(inner_id, labels[j]);
            inner_ids.emplace_back(inner_id, j);
        }
    }
    if (is_initial_build && this->is_cspg_enabled() && !inner_ids.empty()) {
        // 先按论文里的比例一次性采样 routing vectors，再给剩余向量随机分区。
        std::vector<size_t> order(inner_ids.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        std::mt19937 generator(std::random_device{}());
        std::shuffle(order.begin(), order.end(), generator);
        const size_t routing_count = static_cast<size_t>(std::floor(
            static_cast<double>(inner_ids.size()) * static_cast<double>(this->cspg_lambda_)));
        std::uniform_int_distribution<int> partition_dist(0, std::max(1, this->cspg_m_) - 1);
        for (size_t rank = 0; rank < order.size(); ++rank) {
            auto inner_id = inner_ids[order[rank]].first;
            if (rank < routing_count) {
                this->staged_node_partition_[inner_id] = kCspgRoutingPartition;
            } else {
                this->staged_node_partition_[inner_id] = partition_dist(generator);
            }
        }
    }
    for (auto& [inner_id, local_idx] : inner_ids) {
        int level;
        {
            std::scoped_lock label_lock(this->label_lookup_mutex_);
            level = this->get_random_level() - 1;
        }
        const auto* extra_info = extra_infos + local_idx * extra_info_size_;
        const AttributeSet* cur_attr_set = nullptr;
        if (attr_sets != nullptr) {
            cur_attr_set = attr_sets + local_idx;
        }
        if (this->thread_pool_ != nullptr) {
            auto future = this->thread_pool_->GeneralEnqueue(
                add_func, get_data(data, local_idx), level, inner_id, extra_info, cur_attr_set);
            futures.emplace_back(std::move(future));
        } else {
            add_func(get_data(data, local_idx), level, inner_id, extra_info, cur_attr_set);
        }
    }
    if (this->thread_pool_ != nullptr) {
        for (auto& future : futures) {
            future.get();
        }
    }
    return failed_ids;
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter) const {
    return KnnSearch(query, k, parameters, filter, nullptr);
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter,
                  Allocator* allocator) const {
    SearchRequest req;
    req.query_ = query;
    req.topk_ = k;
    req.filter_ = filter;
    req.params_str_ = parameters;
    req.search_allocator_ = allocator;
    return this->SearchWithRequest(req);
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter,
                  Allocator* allocator,
                  IteratorContext*& iter_ctx,
                  bool is_last_filter) const {
    SearchStatistics stats;
    QueryContext ctx{.alloc = allocator_, .stats = &stats};
    if (allocator != nullptr) {
        ctx.alloc = allocator;
    }

    if (GetNumElements() == 0) {
        return DatasetImpl::MakeEmptyDataset();
    }
    int64_t query_dim = query->GetDim();
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    }

    auto params = HGraphSearchParameters::FromJson(parameters);
    auto ef_search_threshold = std::max<int64_t>(AMPLIFICATION_FACTOR * k, 1000);
    CHECK_ARGUMENT(  // NOLINT
        (1 <= params.ef_search) and (params.ef_search <= ef_search_threshold),
        fmt::format("ef_search({}) must in range[1, {}]", params.ef_search, ef_search_threshold));

    std::shared_lock shared_lock(this->global_mutex_);
    // check k
    CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k));
    k = std::min(k, GetNumElements());

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    auto combined_filter = std::make_shared<CombinedFilter>();
    combined_filter->AppendFilter(this->label_table_->GetDeletedIdsFilter());
    if (filter != nullptr) {
        if (params.use_extra_info_filter) {
            combined_filter->AppendFilter(
                std::make_shared<ExtraInfoWrapperFilter>(filter, this->extra_infos_));
        } else {
            combined_filter->AppendFilter(
                std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_));
        }
    }
    FilterPtr ft = nullptr;
    if (not combined_filter->IsEmpty()) {
        ft = combined_filter;
    }

    if (iter_ctx == nullptr) {
        auto cur_count = this->total_count_.load();

        if (cur_count == 0) {
            SearchStatistics stats;
            auto dataset_result = DatasetImpl::MakeEmptyDataset();
            dataset_result->Statistics(stats.Dump());
            return dataset_result;
        }
        auto* new_ctx = new IteratorFilterContext();
        if (auto ret = new_ctx->init(cur_count, params.ef_search, ctx.alloc); not ret.has_value()) {
            delete new_ctx;
            throw vsag::VsagException(ErrorType::INTERNAL_ERROR,
                                      "failed to init IteratorFilterContext");
        }
        iter_ctx = new_ctx;
    }

    auto* iter_filter_ctx = static_cast<IteratorFilterContext*>(iter_ctx);
    auto search_result = DistanceHeap::MakeInstanceBySize<true, false>(ctx.alloc, k);
    const auto* query_data = get_data(query);
    if (is_last_filter) {
        while (!iter_filter_ctx->Empty()) {
            uint32_t cur_inner_id = iter_filter_ctx->GetTopID();
            float cur_dist = iter_filter_ctx->GetTopDist();
            search_result->Push(cur_dist, cur_inner_id);
            iter_filter_ctx->PopDiscard();
        }
    } else {
        InnerSearchParam search_param;
        search_param.ep = this->entry_point_id_;
        search_param.topk = 1;
        search_param.ef = 1;
        search_param.is_inner_id_allowed = nullptr;
        if (iter_filter_ctx->IsFirstUsed()) {
            for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
                auto result = this->search_one_graph(query_data,
                                                     this->route_graphs_[i],
                                                     this->basic_flatten_codes_,
                                                     search_param,
                                                     (VisitedListPtr) nullptr,
                                                     &ctx);
                search_param.ep = result->Top().second;
            }
        }

        search_param.ef = std::max(params.ef_search, k);
        search_param.is_inner_id_allowed = ft;
        search_param.topk = static_cast<int64_t>(search_param.ef);
        search_param.parallel_search_thread_count = params.parallel_search_thread_count;

        search_result = this->search_one_graph(query_data,
                                               this->bottom_graph_,
                                               this->basic_flatten_codes_,
                                               search_param,
                                               iter_filter_ctx,
                                               &ctx);
    }

    if (use_reorder_) {
        this->reorder(
            query_data, this->high_precise_codes_, search_result, k, iter_filter_ctx, ctx);
    }

    while (search_result->Size() > k) {
        auto curr = search_result->Top();
        iter_filter_ctx->AddDiscardNode(curr.first, curr.second);
        search_result->Pop();
    }

    // return an empty dataset directly if searcher returns nothing
    if (search_result->Empty()) {
        return DatasetImpl::MakeEmptyDataset();
    }
    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, ids] = create_fast_dataset(count, ctx.alloc);
    char* extra_infos = nullptr;
    if (extra_info_size_ > 0) {
        extra_infos =
            static_cast<char*>(ctx.alloc->Allocate(extra_info_size_ * search_result->Size()));
        dataset_results->ExtraInfos(extra_infos);
    }
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        ids[j] = this->label_table_->GetLabelById(search_result->Top().second);
        iter_filter_ctx->SetPoint(search_result->Top().second);
        if (extra_infos != nullptr) {
            this->extra_infos_->GetExtraInfoById(search_result->Top().second,
                                                 extra_infos + extra_info_size_ * j);
        }
        search_result->Pop();
    }
    iter_filter_ctx->SetOFFFirstUsed();

    dataset_results->Statistics(stats.Dump());
    return std::move(dataset_results);
}

uint64_t
HGraph::EstimateMemory(uint64_t num_elements) const {
    uint64_t estimate_memory = 0;
    auto block_size = Options::Instance().block_size_limit();
    auto element_count =
        next_multiple_of_power_of_two(num_elements, this->resize_increase_count_bit_);

    auto block_memory_ceil = [](uint64_t memory, uint64_t block_size) -> uint64_t {
        return static_cast<uint64_t>(
            std::ceil(static_cast<double>(memory) / static_cast<double>(block_size)) *
            static_cast<double>(block_size));
    };

    if (this->basic_flatten_codes_->InMemory()) {
        auto base_memory = this->basic_flatten_codes_->code_size_ * element_count;
        estimate_memory += block_memory_ceil(base_memory, block_size);
    }

    if (bottom_graph_->InMemory()) {
        auto bottom_graph_memory =
            (this->bottom_graph_->maximum_degree_ + 1) * sizeof(InnerIdType) * element_count;
        estimate_memory += block_memory_ceil(bottom_graph_memory, block_size);
    }

    if (use_reorder_ && this->high_precise_codes_->InMemory() && not this->ignore_reorder_) {
        auto precise_memory = this->high_precise_codes_->code_size_ * element_count;
        estimate_memory += block_memory_ceil(precise_memory, block_size);
    }

    if (extra_info_size_ > 0 && this->extra_infos_ != nullptr && this->extra_infos_->InMemory()) {
        auto extra_info_memory = this->extra_infos_->ExtraInfoSize() * element_count;
        estimate_memory += block_memory_ceil(extra_info_memory, block_size);
    }

    auto label_map_memory =
        element_count * (sizeof(std::pair<LabelType, InnerIdType>) + 2 * sizeof(void*));
    estimate_memory += label_map_memory;

    auto sparse_graph_memory = (this->mult_ * 0.05 * static_cast<double>(element_count)) *
                               sizeof(InnerIdType) *
                               (static_cast<double>(this->bottom_graph_->maximum_degree_) / 2 + 1);
    estimate_memory += static_cast<uint64_t>(sparse_graph_memory);

    auto other_memory = element_count * (sizeof(LabelType) + sizeof(std::shared_mutex) +
                                         sizeof(std::shared_ptr<std::shared_mutex>));
    estimate_memory += other_memory;

    return estimate_memory;
}

GraphInterfacePtr
HGraph::generate_one_route_graph() {
    return std::make_shared<SparseGraphDataCell>(hierarchical_datacell_param_, this->allocator_);
}

GraphInterfacePtr
HGraph::generate_one_partition_graph() {
    auto graph_param = this->cspg_partition_graph_param_ != nullptr
                           ? this->cspg_partition_graph_param_
                           : this->bottom_graph_param_;
    return GraphInterface::MakeInstance(graph_param, this->common_param_);
}

GraphInterfacePtr
HGraph::generate_one_partition_route_graph() {
    auto graph_param = this->cspg_partition_hierarchical_datacell_param_ != nullptr
                           ? this->cspg_partition_hierarchical_datacell_param_
                           : this->hierarchical_datacell_param_;
    return std::make_shared<SparseGraphDataCell>(graph_param, this->allocator_);
}

void
HGraph::ensure_cspg_partition_graphs() {
    if (not this->is_cspg_enabled()) {
        this->cspg_partition_graphs_.clear();
        return;
    }

    auto partition_count = static_cast<size_t>(std::max(1, this->cspg_m_));
    while (this->cspg_partition_graphs_.size() < partition_count) {
        auto graph = this->generate_one_partition_graph();
        graph->Resize(this->max_capacity_.load());
        this->cspg_partition_graphs_.emplace_back(graph);
    }
}

void
HGraph::ensure_cspg_partition_route_graphs() {
    if (not this->is_cspg_enabled()) {
        this->cspg_partition_route_graphs_.clear();
        this->cspg_partition_route_entry_points_.clear();
        this->cspg_partition_route_entry_levels_.clear();
        return;
    }

    auto partition_count = static_cast<size_t>(std::max(1, this->cspg_m_));
    while (this->cspg_partition_route_graphs_.size() < partition_count) {
        this->cspg_partition_route_graphs_.emplace_back();
    }
    this->cspg_partition_route_entry_points_.resize(partition_count, INVALID_ENTRY_POINT);
    this->cspg_partition_route_entry_levels_.resize(partition_count, -1);
}

void
HGraph::ensure_cspg_partition_route_levels(int partition_id, size_t level_count) {
    this->ensure_cspg_partition_route_graphs();
    if (partition_id < 0 ||
        static_cast<size_t>(partition_id) >= this->cspg_partition_route_graphs_.size()) {
        return;
    }
    auto& route_graphs = this->cspg_partition_route_graphs_[static_cast<size_t>(partition_id)];
    while (route_graphs.size() < level_count) {
        auto graph = this->generate_one_partition_route_graph();
        graph->Resize(this->max_capacity_.load());
        route_graphs.emplace_back(graph);
    }
}

GraphInterfacePtr
HGraph::get_cspg_partition_graph(int partition_id) const {
    if (partition_id < 0 ||
        static_cast<size_t>(partition_id) >= this->cspg_partition_graphs_.size()) {
        return nullptr;
    }
    return this->cspg_partition_graphs_[partition_id];
}

GraphInterfacePtr
HGraph::get_cspg_partition_route_graph(int partition_id, int level) const {
    if (partition_id < 0 || level < 0 ||
        static_cast<size_t>(partition_id) >= this->cspg_partition_route_graphs_.size()) {
        return nullptr;
    }
    const auto& route_graphs =
        this->cspg_partition_route_graphs_[static_cast<size_t>(partition_id)];
    if (static_cast<size_t>(level) >= route_graphs.size()) {
        return nullptr;
    }
    return route_graphs[static_cast<size_t>(level)];
}

size_t
HGraph::get_cspg_partition_route_level_count(int partition_id) const {
    if (partition_id < 0 ||
        static_cast<size_t>(partition_id) >= this->cspg_partition_route_graphs_.size()) {
        return 0;
    }
    return this->cspg_partition_route_graphs_[static_cast<size_t>(partition_id)].size();
}

std::vector<int>
HGraph::get_cspg_target_partitions(int assigned_partition) const {
    std::vector<int> target_partitions;
    if (assigned_partition == kCspgRoutingPartition) {
        target_partitions.reserve(static_cast<size_t>(std::max(1, this->cspg_m_)));
        for (int partition = 0; partition < std::max(1, this->cspg_m_); ++partition) {
            target_partitions.emplace_back(partition);
        }
        return target_partitions;
    }
    if (assigned_partition >= 0 && assigned_partition < std::max(1, this->cspg_m_)) {
        target_partitions.emplace_back(assigned_partition);
    }
    return target_partitions;
}

InnerIdType
HGraph::find_cspg_partition_route_entry(int partition_id, int* level) const {
    if (level != nullptr) {
        *level = -1;
    }
    if (partition_id < 0 ||
        static_cast<size_t>(partition_id) >= this->cspg_partition_route_graphs_.size()) {
        return INVALID_ENTRY_POINT;
    }

    const auto& route_graphs =
        this->cspg_partition_route_graphs_[static_cast<size_t>(partition_id)];
    for (int current_level = static_cast<int>(route_graphs.size()) - 1; current_level >= 0;
         --current_level) {
        const auto& graph = route_graphs[static_cast<size_t>(current_level)];
        if (graph == nullptr || graph->TotalCount() == 0) {
            continue;
        }
        if (static_cast<size_t>(partition_id) < this->cspg_partition_route_entry_points_.size()) {
            auto entry =
                this->cspg_partition_route_entry_points_[static_cast<size_t>(partition_id)];
            if (entry != INVALID_ENTRY_POINT && graph->CheckIdExists(entry)) {
                if (level != nullptr) {
                    *level = current_level;
                }
                return entry;
            }
        }
        auto ids = graph->GetIds();
        if (!ids.empty()) {
            if (level != nullptr) {
                *level = current_level;
            }
            return ids[0];
        }
    }
    return INVALID_ENTRY_POINT;
}

void
HGraph::rebuild_cspg_partition_route_entries() {
    if (not this->is_cspg_enabled()) {
        this->cspg_partition_route_entry_points_.clear();
        this->cspg_partition_route_entry_levels_.clear();
        return;
    }

    auto partition_count = static_cast<size_t>(std::max(1, this->cspg_m_));
    this->cspg_partition_route_entry_points_.assign(partition_count, INVALID_ENTRY_POINT);
    this->cspg_partition_route_entry_levels_.assign(partition_count, -1);
    for (size_t partition_id = 0; partition_id < partition_count; ++partition_id) {
        int level = -1;
        auto entry = this->find_cspg_partition_route_entry(static_cast<int>(partition_id), &level);
        this->cspg_partition_route_entry_points_[partition_id] = entry;
        this->cspg_partition_route_entry_levels_[partition_id] = level;
    }
}

InnerIdType
HGraph::choose_any_cspg_entry_point() const {
    for (auto entry : this->cspg_partition_route_entry_points_) {
        if (entry != INVALID_ENTRY_POINT) {
            return entry;
        }
    }
    for (auto entry : this->cspg_partition_entry_points_) {
        if (entry != INVALID_ENTRY_POINT) {
            return entry;
        }
    }
    for (size_t partition_id = 0; partition_id < this->cspg_partition_graphs_.size();
         ++partition_id) {
        auto graph = this->cspg_partition_graphs_[partition_id];
        if (graph == nullptr) {
            continue;
        }
        auto ids = graph->GetIds();
        if (!ids.empty()) {
            return ids[0];
        }
    }
    return INVALID_ENTRY_POINT;
}

template <InnerSearchMode mode>
DistHeapPtr
HGraph::search_one_graph(const void* query,
                         const GraphInterfacePtr& graph,
                         const FlattenInterfacePtr& flatten,
                         InnerSearchParam& inner_search_param,
                         const VisitedListPtr& vt,
                         QueryContext* ctx) const {
    bool new_visited_list = vt == nullptr;
    VisitedListPtr visited_list;
    if (new_visited_list) {
        visited_list = this->pool_->TakeOne();
    } else {
        visited_list = vt;
        visited_list->Reset();
    }
    DistHeapPtr result = nullptr;
    if (inner_search_param.parallel_search_thread_count > 1) {
        result = this->parallel_searcher_->Search(
            graph, flatten, visited_list, query, inner_search_param);
    } else {
        result = this->searcher_->Search(
            graph, flatten, visited_list, query, inner_search_param, this->label_table_, ctx);
    }
    if (new_visited_list) {
        this->pool_->ReturnOne(visited_list);
    }
    return result;
}

template <InnerSearchMode mode>
DistHeapPtr
HGraph::search_one_graph(const void* query,
                         const GraphInterfacePtr& graph,
                         const FlattenInterfacePtr& flatten,
                         InnerSearchParam& inner_search_param,
                         IteratorFilterContext* iter_ctx,
                         QueryContext* ctx) const {
    auto visited_list = this->pool_->TakeOne();
    auto result = this->searcher_->Search(
        graph, flatten, visited_list, query, inner_search_param, iter_ctx, ctx);
    this->pool_->ReturnOne(visited_list);
    return result;
}

DatasetPtr
HGraph::RangeSearch(const DatasetPtr& query,
                    float radius,
                    const std::string& parameters,
                    const FilterPtr& filter,
                    int64_t limited_size) const {
    SearchStatistics stats;
    QueryContext ctx{.stats = &stats};

    auto combined_filter = std::make_shared<CombinedFilter>();
    combined_filter->AppendFilter(this->label_table_->GetDeletedIdsFilter());
    if (filter != nullptr) {
        combined_filter->AppendFilter(
            std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_));
    }
    FilterPtr ft = nullptr;
    if (not combined_filter->IsEmpty()) {
        ft = combined_filter;
    }

    int64_t query_dim = query->GetDim();
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    }
    // check radius
    CHECK_ARGUMENT(radius >= 0, fmt::format("radius({}) must be greater equal than 0", radius))

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    // check limited_size
    CHECK_ARGUMENT(limited_size != 0,
                   fmt::format("limited_size({}) must not be equal to 0", limited_size));

    InnerSearchParam search_param;
    search_param.ep = this->entry_point_id_;
    search_param.topk = 1;
    search_param.ef = 1;
    const auto* raw_query = get_data(query);
    for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
        auto result = this->search_one_graph(raw_query,
                                             this->route_graphs_[i],
                                             this->basic_flatten_codes_,
                                             search_param,
                                             (VisitedListPtr) nullptr,
                                             &ctx);
        search_param.ep = result->Top().second;
    }

    auto params = HGraphSearchParameters::FromJson(parameters);

    CHECK_ARGUMENT((1 <= params.ef_search) and (params.ef_search <= 1000),  // NOLINT
                   fmt::format("ef_search({}) must in range[1, 1000]", params.ef_search));
    search_param.ef = std::max(params.ef_search, limited_size);
    search_param.is_inner_id_allowed = ft;
    search_param.radius = radius;
    search_param.search_mode = RANGE_SEARCH;
    search_param.consider_duplicate = true;
    search_param.range_search_limit_size = static_cast<int>(limited_size);
    search_param.parallel_search_thread_count = params.parallel_search_thread_count;

    auto search_result = this->search_one_graph(raw_query,
                                                this->bottom_graph_,
                                                this->basic_flatten_codes_,
                                                search_param,
                                                (VisitedListPtr) nullptr,
                                                &ctx);

    if (use_reorder_) {
        this->reorder(
            raw_query, this->high_precise_codes_, search_result, limited_size, nullptr, ctx);
    }

    if (limited_size > 0) {
        while (search_result->Size() > limited_size) {
            search_result->Pop();
        }
    }

    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, ids] = create_fast_dataset(count, allocator_);
    char* extra_infos = nullptr;
    if (extra_info_size_ > 0) {
        extra_infos =
            static_cast<char*>(allocator_->Allocate(extra_info_size_ * search_result->Size()));
        dataset_results->ExtraInfos(extra_infos);
    }
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        ids[j] = this->label_table_->GetLabelById(search_result->Top().second);
        if (extra_infos != nullptr) {
            this->extra_infos_->GetExtraInfoById(search_result->Top().second,
                                                 extra_infos + extra_info_size_ * j);
        }
        search_result->Pop();
    }

    dataset_results->Statistics(stats.Dump());
    return std::move(dataset_results);
}

void
HGraph::serialize_basic_info_v0_14(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, this->use_reorder_);
    StreamWriter::WriteObj(writer, this->dim_);
    StreamWriter::WriteObj(writer, this->metric_);
    uint64_t max_level = this->route_graphs_.size();
    StreamWriter::WriteObj(writer, max_level);
    StreamWriter::WriteObj(writer, this->entry_point_id_);
    StreamWriter::WriteObj(writer, this->ef_construct_);
    StreamWriter::WriteObj(writer, this->mult_);
    auto capacity = this->max_capacity_.load();
    StreamWriter::WriteObj(writer, capacity);
    StreamWriter::WriteVector(writer, this->label_table_->label_table_);

    uint64_t size = this->label_table_->label_remap_.size();
    StreamWriter::WriteObj(writer, size);
    for (const auto& pair : this->label_table_->label_remap_) {
        auto key = pair.first;
        StreamWriter::WriteObj(writer, key);
        StreamWriter::WriteObj(writer, pair.second);
    }
}

void
HGraph::deserialize_basic_info_v0_14(StreamReader& reader) {
    StreamReader::ReadObj(reader, this->use_reorder_);
    StreamReader::ReadObj(reader, this->dim_);
    StreamReader::ReadObj(reader, this->metric_);
    uint64_t max_level;
    StreamReader::ReadObj(reader, max_level);
    for (uint64_t i = 0; i < max_level; ++i) {
        this->route_graphs_.emplace_back(this->generate_one_route_graph());
    }
    StreamReader::ReadObj(reader, this->entry_point_id_);
    StreamReader::ReadObj(reader, this->ef_construct_);
    StreamReader::ReadObj(reader, this->mult_);
    InnerIdType capacity;
    StreamReader::ReadObj(reader, capacity);
    this->max_capacity_.store(capacity);
    StreamReader::ReadVector(reader, this->label_table_->label_table_);

    uint64_t size;
    StreamReader::ReadObj(reader, size);
    for (uint64_t i = 0; i < size; ++i) {
        LabelType key;
        StreamReader::ReadObj(reader, key);
        InnerIdType value;
        StreamReader::ReadObj(reader, value);
        this->label_table_->label_remap_.emplace(key, value);
    }
    // Restore total_count from label_remap size
    this->label_table_->total_count_.store(static_cast<int64_t>(size));
}

#define TO_JSON_BASE64(json_obj, var) json_obj[#var].SetString(base64_encode_obj(this->var##_));

JsonType
HGraph::serialize_basic_info() const {
    JsonType jsonify_basic_info;
    const bool write_cspg_partition = this->is_cspg_enabled();
    // These flags indicate whether the serialized payload contains the CSPG graph sections.
    // The sections are always written when CSPG is enabled, even if some of them are empty.
    const bool write_cspg_partition_graphs = write_cspg_partition;
    const bool write_cspg_partition_route_graphs = write_cspg_partition;
    jsonify_basic_info["use_reorder"].SetBool(this->use_reorder_);
    jsonify_basic_info["dim"].SetInt(this->dim_);
    jsonify_basic_info["metric"].SetInt(static_cast<int64_t>(this->metric_));
    jsonify_basic_info["entry_point_id"].SetInt(this->entry_point_id_);
    jsonify_basic_info["ef_construct"].SetInt(this->ef_construct_);
    jsonify_basic_info["extra_info_size"].SetInt(this->extra_info_size_);
    jsonify_basic_info["data_type"].SetInt(static_cast<int64_t>(this->data_type_));
    // 仅在显式启用 CSPG 时写入分区相关数据，避免污染普通 HGraph 序列化格式。
    jsonify_basic_info["cspg_partition"].SetBool(write_cspg_partition);
    jsonify_basic_info["cspg_partition_graphs"].SetBool(write_cspg_partition_graphs);
    jsonify_basic_info["cspg_partition_route_graphs"].SetBool(write_cspg_partition_route_graphs);
    if (write_cspg_partition && !this->cspg_partition_entry_points_.empty()) {
        std::vector<int32_t> serialized_partition_entries;
        serialized_partition_entries.reserve(this->cspg_partition_entry_points_.size());
        for (const auto entry_point : this->cspg_partition_entry_points_) {
            serialized_partition_entries.emplace_back(static_cast<int32_t>(entry_point));
        }
        jsonify_basic_info["cspg_partition_entry_points"].SetVector(
            std::move(serialized_partition_entries));
    }
    if (this->cspg_partition_graph_param_ != nullptr) {
        switch (this->cspg_partition_graph_param_->graph_storage_type_) {
            case GraphStorageTypes::GRAPH_STORAGE_TYPE_SPARSE:
                jsonify_basic_info["cspg_partition_graph_storage_type"].SetString("sparse");
                break;
            case GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_COMPRESSED:
                jsonify_basic_info["cspg_partition_graph_storage_type"].SetString("compressed");
                break;
            case GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT:
            default:
                jsonify_basic_info["cspg_partition_graph_storage_type"].SetString("flat");
                break;
        }
    }
    // logger::debug("mult: {}", this->mult_);
    TO_JSON_BASE64(jsonify_basic_info, mult);
    jsonify_basic_info["max_capacity"].SetInt(this->max_capacity_.load());
    jsonify_basic_info["max_level"].SetInt(this->route_graphs_.size());
    jsonify_basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());

    return jsonify_basic_info;
}

#define FROM_JSON(json_obj, var, type)                   \
    do {                                                 \
        if ((json_obj).Contains(#var)) {                 \
            this->var##_ = (json_obj)[#var].Get##type(); \
        }                                                \
    } while (0)

#define FROM_JSON_BASE64(json_obj, var) \
    base64_decode_obj((json_obj)[#var].GetString(), this->var##_);

void
HGraph::deserialize_basic_info(const JsonType& jsonify_basic_info) {
    logger::debug("jsonify_basic_info: {}", jsonify_basic_info.Dump());
    FROM_JSON(jsonify_basic_info, use_reorder, Bool);
    FROM_JSON(jsonify_basic_info, dim, Int);
    if (jsonify_basic_info.Contains("metric")) {
        this->metric_ = static_cast<MetricType>(jsonify_basic_info["metric"].GetInt());
    }
    FROM_JSON(jsonify_basic_info, entry_point_id, Int);
    FROM_JSON(jsonify_basic_info, ef_construct, Int);
    FROM_JSON(jsonify_basic_info, extra_info_size, Int);
    if (jsonify_basic_info.Contains("data_type")) {
        this->data_type_ = static_cast<DataTypes>(jsonify_basic_info["data_type"].GetInt());
    }
    FROM_JSON_BASE64(jsonify_basic_info, mult);
    // logger::debug("mult: {}", this->mult_);
    this->max_capacity_.store(jsonify_basic_info["max_capacity"].GetInt());

    auto max_level = jsonify_basic_info["max_level"].GetInt();
    for (int64_t i = 0; i < max_level; ++i) {
        this->route_graphs_.emplace_back(this->generate_one_route_graph());
    }
    if (jsonify_basic_info.Contains(INDEX_PARAM)) {
        std::string index_param_string = jsonify_basic_info[INDEX_PARAM].GetString();
        HGraphParameterPtr index_param = std::make_shared<HGraphParameter>();
        index_param->data_type = this->data_type_;
        index_param->FromString(index_param_string);
        if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
            auto message = fmt::format("HGraph index parameter not match, current: {}, new: {}",
                                       this->create_param_ptr_->ToString(),
                                       index_param->ToString());
            logger::error(message);
            throw VsagException(ErrorType::INVALID_ARGUMENT, message);
        }
    }
}

void
HGraph::serialize_label_info(StreamWriter& writer) const {
    if (this->label_table_->CompressDuplicateData()) {
        this->label_table_->Serialize(writer);
        return;
    }
    StreamWriter::WriteVector(writer, this->label_table_->label_table_);
    uint64_t size = this->label_table_->label_remap_.size();
    StreamWriter::WriteObj(writer, size);
    for (const auto& pair : this->label_table_->label_remap_) {
        auto key = pair.first;
        StreamWriter::WriteObj(writer, key);
        StreamWriter::WriteObj(writer, pair.second);
    }
}

void
HGraph::deserialize_label_info(StreamReader& reader) const {
    if (this->label_table_->CompressDuplicateData()) {
        this->label_table_->Deserialize(reader);
        return;
    }
    StreamReader::ReadVector(reader, this->label_table_->label_table_);
    uint64_t size;
    StreamReader::ReadObj(reader, size);
    for (uint64_t i = 0; i < size; ++i) {
        LabelType key;
        StreamReader::ReadObj(reader, key);
        InnerIdType value;
        StreamReader::ReadObj(reader, value);
        this->label_table_->label_remap_.emplace(key, value);
    }
    // Restore total_count from label_remap size (same as number of valid elements)
    this->label_table_->total_count_.store(static_cast<int64_t>(size));
}

void
HGraph::Serialize(StreamWriter& writer) const {
    if (this->ignore_reorder_) {
        this->use_reorder_ = false;
    }

    // FIXME(wxyu): this option is used for special purposes, like compatibility testing
    if (this->use_old_serial_format_) {
        this->serialize_basic_info_v0_14(writer);
        this->basic_flatten_codes_->Serialize(writer);
        this->bottom_graph_->Serialize(writer);
        if (this->use_reorder_) {
            this->high_precise_codes_->Serialize(writer);
        }
        for (const auto& route_graph : this->route_graphs_) {
            route_graph->Serialize(writer);
        }
        if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
            this->extra_infos_->Serialize(writer);
        }
        if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
            this->attr_filter_index_->Serialize(writer);
        }
        return;
    }

    this->serialize_label_info(writer);
    this->basic_flatten_codes_->Serialize(writer);
    this->bottom_graph_->Serialize(writer);
    if (this->is_cspg_enabled()) {
        // 写入 CSPG 分区数组，确保显式开启时索引复现一致。
        StreamWriter::WriteVector(writer, this->node_partition_);
        uint64_t partition_graph_count = this->cspg_partition_graphs_.size();
        StreamWriter::WriteObj(writer, partition_graph_count);
        for (const auto& partition_graph : this->cspg_partition_graphs_) {
            partition_graph->Serialize(writer);
        }
        uint64_t partition_route_graph_count = this->cspg_partition_route_graphs_.size();
        StreamWriter::WriteObj(writer, partition_route_graph_count);
        for (const auto& route_graphs : this->cspg_partition_route_graphs_) {
            uint64_t level_count = route_graphs.size();
            StreamWriter::WriteObj(writer, level_count);
            for (const auto& route_graph : route_graphs) {
                route_graph->Serialize(writer);
            }
        }
    }
    if (this->use_reorder_) {
        this->high_precise_codes_->Serialize(writer);
    }
    for (const auto& route_graph : this->route_graphs_) {
        route_graph->Serialize(writer);
    }
    if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
        this->extra_infos_->Serialize(writer);
    }
    if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        this->attr_filter_index_->Serialize(writer);
    }
    if (create_new_raw_vector_) {
        this->raw_vector_->Serialize(writer);
    }

    // serialize footer (introduced since v0.15)
    auto jsonify_basic_info = this->serialize_basic_info();
    auto metadata = std::make_shared<Metadata>();
    metadata->Set(BASIC_INFO, jsonify_basic_info);
    logger::debug(jsonify_basic_info.Dump());

    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

void
HGraph::Deserialize(StreamReader& reader) {
    // try to deserialize footer (only in new version)
    auto footer = Footer::Parse(reader);

    if (footer == nullptr) {  // old format, DON'T EDIT, remove in the future
        logger::debug("parse with v0.14 version format");

        this->deserialize_basic_info_v0_14(reader);

        this->basic_flatten_codes_->Deserialize(reader);
        this->bottom_graph_->Deserialize(reader);
        if (this->use_reorder_) {
            this->high_precise_codes_->Deserialize(reader);
        }

        for (auto& route_graph : this->route_graphs_) {
            route_graph->Deserialize(reader);
        }
        auto new_size = max_capacity_.load();
        this->neighbors_mutex_->Resize(new_size);

        pool_ = std::make_shared<VisitedListPool>(1, allocator_, new_size, allocator_);
        // 老版本序列化没有分区信息，默认全部设为路由点占位。
        this->node_partition_.assign(new_size, kCspgRoutingPartition);
        this->staged_node_partition_.assign(new_size, kCspgUnassignedPartition);

        if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
            this->extra_infos_->Deserialize(reader);
        }
        this->total_count_ = this->basic_flatten_codes_->TotalCount();

        if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
            this->attr_filter_index_->Deserialize(reader);
        }
    } else {  // create like `else if ( ver in [v0.15, v0.17] )` here if need in the future
        logger::debug("parse with new version format");

        BufferStreamReader buffer_reader(
            &reader, std::numeric_limits<uint64_t>::max(), this->allocator_);

        auto metadata = footer->GetMetadata();
        // metadata should NOT be nullptr if footer is not nullptr
        auto basic_info = metadata->Get(BASIC_INFO);
        this->deserialize_basic_info(basic_info);
        this->deserialize_label_info(buffer_reader);

        this->basic_flatten_codes_->Deserialize(buffer_reader);
        this->bottom_graph_->Deserialize(buffer_reader);
        const bool has_cspg_partition =
            basic_info.Contains("cspg_partition") && basic_info["cspg_partition"].GetBool();
        const bool has_cspg_partition_route_graphs =
            basic_info.Contains("cspg_partition_route_graphs") &&
            basic_info["cspg_partition_route_graphs"].GetBool();
        if (has_cspg_partition) {
            auto hgraph_param = std::dynamic_pointer_cast<HGraphParameter>(this->create_param_ptr_);
            const auto cspg_partition_max_degree =
                hgraph_param != nullptr
                    ? static_cast<uint64_t>(hgraph_param->ResolveCspgPartitionMaxDegree())
                    : this->bottom_graph_param_->max_degree_;
            bool use_sparse_partition_graph_storage = true;
            if (basic_info.Contains("cspg_partition_graph_storage_type")) {
                const auto storage_type =
                    basic_info["cspg_partition_graph_storage_type"].GetString();
                use_sparse_partition_graph_storage = storage_type == "sparse";
            } else {
                // Legacy CSPG indexes serialized partition graphs with the bottom graph storage.
                use_sparse_partition_graph_storage = false;
            }
            this->cspg_partition_graph_param_ = create_cspg_partition_graph_param(
                this->bottom_graph_param_,
                cspg_partition_max_degree,
                use_sparse_partition_graph_storage);
        }
        // 根据元数据判断是否包含 CSPG 分区数组。
        if (has_cspg_partition) {
            StreamReader::ReadVector(buffer_reader, this->node_partition_);
            uint64_t partition_graph_count = 0;
            StreamReader::ReadObj(buffer_reader, partition_graph_count);
            this->cspg_partition_graphs_.clear();
            for (uint64_t i = 0; i < partition_graph_count; ++i) {
                auto partition_graph = this->generate_one_partition_graph();
                partition_graph->Deserialize(buffer_reader);
                this->cspg_partition_graphs_.emplace_back(partition_graph);
            }
            if (has_cspg_partition_route_graphs) {
                uint64_t partition_route_graph_count = 0;
                StreamReader::ReadObj(buffer_reader, partition_route_graph_count);
                this->cspg_partition_route_graphs_.clear();
                this->cspg_partition_route_graphs_.resize(partition_route_graph_count);
                for (uint64_t partition_id = 0; partition_id < partition_route_graph_count;
                     ++partition_id) {
                    uint64_t level_count = 0;
                    StreamReader::ReadObj(buffer_reader, level_count);
                    auto& route_graphs = this->cspg_partition_route_graphs_[partition_id];
                    route_graphs.clear();
                    route_graphs.reserve(level_count);
                    for (uint64_t level = 0; level < level_count; ++level) {
                        auto route_graph = this->generate_one_partition_route_graph();
                        route_graph->Deserialize(buffer_reader);
                        route_graphs.emplace_back(route_graph);
                    }
                }
            } else {
                this->cspg_partition_route_graphs_.clear();
            }
        } else {
            this->node_partition_.assign(this->max_capacity_.load(), kCspgUnassignedPartition);
            this->cspg_partition_graphs_.clear();
            this->cspg_partition_route_graphs_.clear();
        }
        this->staged_node_partition_.assign(this->max_capacity_.load(), kCspgUnassignedPartition);
        if (this->is_cspg_enabled()) {
            if (basic_info.Contains("cspg_partition_entry_points")) {
                auto serialized_partition_entries =
                    basic_info["cspg_partition_entry_points"].GetVector();
                this->cspg_partition_entry_points_.assign(std::max(1, this->cspg_m_),
                                                          INVALID_ENTRY_POINT);
                for (size_t i = 0;
                     i < serialized_partition_entries.size() &&
                     i < this->cspg_partition_entry_points_.size();
                     ++i) {
                    this->cspg_partition_entry_points_[i] =
                        static_cast<InnerIdType>(serialized_partition_entries[i]);
                }
            } else {
                this->cspg_partition_entry_points_ =
                    BuildCspgPartitionEntryPoints(this->node_partition_, this->cspg_m_);
            }
            this->ensure_cspg_partition_route_graphs();
            this->rebuild_cspg_partition_route_entries();
        } else {
            this->cspg_partition_entry_points_.clear();
            this->cspg_partition_route_entry_points_.clear();
            this->cspg_partition_route_entry_levels_.clear();
        }
        if (this->use_reorder_) {
            this->high_precise_codes_->Deserialize(buffer_reader);
        }

        for (auto& route_graph : this->route_graphs_) {
            route_graph->Deserialize(buffer_reader);
        }
        auto new_size = max_capacity_.load();
        this->neighbors_mutex_->Resize(new_size);

        pool_ = std::make_shared<VisitedListPool>(1, allocator_, new_size, allocator_);

        if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
            this->extra_infos_->Deserialize(buffer_reader);
        }
        this->total_count_ = this->basic_flatten_codes_->TotalCount();

        if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
            this->attr_filter_index_->Deserialize(buffer_reader);
        }

        if (create_new_raw_vector_) {
            this->raw_vector_->Deserialize(buffer_reader);
        }
        if (this->raw_vector_ != nullptr) {
            this->has_raw_vector_ = true;
        }
    }
    this->cal_memory_usage();

    // post serialize procedure
    if (use_elp_optimizer_) {
        elp_optimize();
    }
}

std::string
HGraph::GetMemoryUsageDetail() const {
    JsonType memory_usage;
    if (this->ignore_reorder_) {
        this->use_reorder_ = false;
    }
    memory_usage["basic_flatten_codes"].SetInt(this->basic_flatten_codes_->CalcSerializeSize());
    memory_usage["bottom_graph"].SetInt(this->bottom_graph_->CalcSerializeSize());
    if (this->use_reorder_) {
        memory_usage["high_precise_codes"].SetInt(this->high_precise_codes_->CalcSerializeSize());
    }
    uint64_t route_graph_size = 0;
    for (const auto& route_graph : this->route_graphs_) {
        route_graph_size += route_graph->CalcSerializeSize();
    }
    memory_usage["route_graph"].SetInt(route_graph_size);
    uint64_t cspg_partition_graph_size = 0;
    for (const auto& graph : this->cspg_partition_graphs_) {
        if (graph != nullptr) {
            cspg_partition_graph_size += graph->CalcSerializeSize();
        }
    }
    memory_usage["cspg_partition_graph"].SetInt(cspg_partition_graph_size);
    uint64_t cspg_partition_route_graph_size = 0;
    for (const auto& route_graphs : this->cspg_partition_route_graphs_) {
        for (const auto& graph : route_graphs) {
            if (graph != nullptr) {
                cspg_partition_route_graph_size += graph->CalcSerializeSize();
            }
        }
    }
    memory_usage["cspg_partition_route_graph"].SetInt(cspg_partition_route_graph_size);
    if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
        memory_usage["extra_infos"].SetInt(this->extra_infos_->CalcSerializeSize());
    }
    memory_usage["__total_size__"].SetInt(this->CalSerializeSize());
    return memory_usage.Dump();
}

float
HGraph::CalcDistanceById(const float* query, int64_t id, bool calculate_precise_distance) const {
    auto flat = this->basic_flatten_codes_;
    if (use_reorder_ && calculate_precise_distance) {
        flat = this->high_precise_codes_;
    }
    if (create_new_raw_vector_ && calculate_precise_distance) {
        flat = this->raw_vector_;
    }
    return InnerIndexInterface::calc_distance_by_id(query, id, flat);
}

DatasetPtr
HGraph::CalDistanceById(const float* query,
                        const int64_t* ids,
                        int64_t count,
                        bool calculate_precise_distance) const {
    auto flat = this->basic_flatten_codes_;
    if (use_reorder_ && calculate_precise_distance) {
        flat = this->high_precise_codes_;
    }
    if (create_new_raw_vector_ && calculate_precise_distance) {
        flat = this->raw_vector_;
    }
    return InnerIndexInterface::cal_distance_by_id(query, ids, count, flat);
}

std::pair<int64_t, int64_t>
HGraph::GetMinAndMaxId() const {
    int64_t min_id = INT64_MAX;
    int64_t max_id = INT64_MIN;
    std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
    if (this->total_count_ == 0) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Label map size is zero");
    }
    for (int i = 0; i < this->total_count_; ++i) {
        if (this->label_table_->IsRemoved(i)) {
            continue;
        }
        auto label = this->label_table_->GetLabelById(i);
        max_id = std::max(label, max_id);
        min_id = std::min(label, min_id);
    }
    return {min_id, max_id};
}

void
HGraph::add_one_point(const void* data, int level, InnerIdType inner_id) {
    {
        std::shared_lock add_lock(add_mutex_);
        this->basic_flatten_codes_->InsertVector(data, inner_id);
        if (use_reorder_) {
            this->high_precise_codes_->InsertVector(data, inner_id);
        }
        if (create_new_raw_vector_) {
            raw_vector_->InsertVector(data, inner_id);
        }
    }

    // CSPG：优先使用批量建图阶段预采样的分区；若没有预采样，则沿用随机分配。
    if (this->is_cspg_enabled()) {
        int assigned_partition = kCspgUnassignedPartition;
        if (static_cast<size_t>(inner_id) < this->staged_node_partition_.size()) {
            assigned_partition = this->staged_node_partition_[inner_id];
            this->staged_node_partition_[inner_id] = kCspgUnassignedPartition;
        }
        if (assigned_partition == kCspgUnassignedPartition) {
            thread_local std::mt19937 local_gen(std::random_device{}());
            std::uniform_real_distribution<float> routing_dist(0.0F, 1.0F);
            int partition_count = std::max(1, this->cspg_m_);
            std::uniform_int_distribution<int> partition_dist(0, partition_count - 1);
            float rand_val = routing_dist(local_gen);
            assigned_partition =
                rand_val < this->cspg_lambda_ ? kCspgRoutingPartition : partition_dist(local_gen);
        }
        node_partition_[inner_id] = assigned_partition;
    }
    // ===================================

    bool enable_cspg = this->is_cspg_enabled() && !this->cspg_partition_graphs_.empty();
    auto target_partitions = enable_cspg
                                 ? this->get_cspg_target_partitions(node_partition_[inner_id])
                                 : std::vector<int>{};
    bool need_full_lock =
        this->entry_point_id_ == INVALID_ENTRY_POINT ||
        (not enable_cspg && (level >= static_cast<int>(this->route_graphs_.size()) ||
                             bottom_graph_->TotalCount() == 0));
    if (enable_cspg && this->entry_point_id_ == INVALID_ENTRY_POINT) {
        need_full_lock = true;
    }

    std::unique_lock add_lock(add_mutex_);
    if (enable_cspg) {
        if (this->cspg_partition_entry_points_.empty()) {
            this->cspg_partition_entry_points_.assign(std::max(1, this->cspg_m_),
                                                      INVALID_ENTRY_POINT);
        }
        if (node_partition_[inner_id] == kCspgRoutingPartition) {
            for (auto& entry_point : this->cspg_partition_entry_points_) {
                if (entry_point == INVALID_ENTRY_POINT) {
                    entry_point = inner_id;
                }
            }
        } else if (node_partition_[inner_id] >= 0 &&
                   static_cast<size_t>(node_partition_[inner_id]) <
                       this->cspg_partition_entry_points_.size()) {
            auto& entry_point = this->cspg_partition_entry_points_[node_partition_[inner_id]];
            if (entry_point == INVALID_ENTRY_POINT) {
                entry_point = inner_id;
            }
        }
    }
    if (need_full_lock) {
        auto old_route_graph_count = this->route_graphs_.size();
        bool created_new_route_graph = false;
        std::scoped_lock<std::shared_mutex> wlock(this->global_mutex_);
        if (!enable_cspg) {
            // level maybe a negative number(-1)
            for (auto j = static_cast<int>(this->route_graphs_.size()); j <= level; ++j) {
                this->route_graphs_.emplace_back(this->generate_one_route_graph());
                created_new_route_graph = true;
            }
        }
        auto insert_success = this->graph_add_one(data, level, inner_id);
        if (insert_success && enable_cspg) {
            this->entry_point_id_ = this->choose_any_cspg_entry_point();
        } else if (insert_success && (this->entry_point_id_ == INVALID_ENTRY_POINT ||
                                      level >= static_cast<int>(old_route_graph_count))) {
            entry_point_id_ = inner_id;
        }
        if (not insert_success && created_new_route_graph) {
            this->route_graphs_.pop_back();
        }
        add_lock.unlock();
    } else {
        add_lock.unlock();
        std::shared_lock rlock(this->global_mutex_);
        this->graph_add_one(data, level, inner_id);
    }
}

bool
HGraph::graph_add_one(const void* data, int level, InnerIdType inner_id) {
    DistHeapPtr result = nullptr;
    InnerSearchParam param;
    param.topk = 1;
    param.ep = this->entry_point_id_;
    param.ef = 1;
    param.is_inner_id_allowed = nullptr;

    auto flatten_codes = basic_flatten_codes_;
    if (use_reorder_ and not build_by_base_) {
        flatten_codes = high_precise_codes_;
    }

    bool enable_cspg = this->is_cspg_enabled() && !this->cspg_partition_graphs_.empty();
    if (!enable_cspg) {
        for (auto j = this->route_graphs_.size() - 1; j > level; --j) {
            result = search_one_graph(
                data, route_graphs_[j], flatten_codes, param, (VisitedListPtr) nullptr, nullptr);
            param.ep = result->Top().second;
        }

        param.ef = this->ef_construct_;
        param.topk = static_cast<int64_t>(ef_construct_);
        if (this->label_table_->CompressDuplicateData()) {
            param.find_duplicate = true;
        }

        if (bottom_graph_->TotalCount() != 0) {
            result = search_one_graph(
                data, this->bottom_graph_, flatten_codes, param, (VisitedListPtr) nullptr, nullptr);
            if (this->label_table_->CompressDuplicateData() && param.duplicate_id >= 0) {
                std::unique_lock lock(this->label_lookup_mutex_);
                label_table_->SetDuplicateId(static_cast<InnerIdType>(param.duplicate_id),
                                             inner_id);
                return false;
            }
            auto filtered_result = std::make_shared<StandardHeap<true, false>>(allocator_, -1);

            while (not result->Empty()) {
                auto [dist, id] = result->Top();
                result->Pop();
                if (id != inner_id) {
                    filtered_result->Push(dist, id);
                }
            }

            LockGuard cur_lock(neighbors_mutex_, inner_id);
            mutually_connect_new_element(inner_id,
                                         filtered_result,
                                         this->bottom_graph_,
                                         flatten_codes,
                                         neighbors_mutex_,
                                         allocator_,
                                         alpha_);
        } else {
            LockGuard cur_lock(neighbors_mutex_, inner_id);
            bottom_graph_->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
        }

        for (int64_t j = 0; j <= level; ++j) {
            if (route_graphs_[j]->TotalCount() != 0) {
                result = search_one_graph(data,
                                          route_graphs_[j],
                                          flatten_codes,
                                          param,
                                          (VisitedListPtr) nullptr,
                                          nullptr);
                auto filtered_result = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
                while (not result->Empty()) {
                    auto [dist, id] = result->Top();
                    result->Pop();
                    if (id != inner_id) {
                        filtered_result->Push(dist, id);
                    }
                }
                LockGuard cur_lock(neighbors_mutex_, inner_id);
                mutually_connect_new_element(inner_id,
                                             filtered_result,
                                             route_graphs_[j],
                                             flatten_codes,
                                             neighbors_mutex_,
                                             allocator_,
                                             alpha_);
            } else {
                LockGuard cur_lock(neighbors_mutex_, inner_id);
                route_graphs_[j]->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
            }
        }
        return true;
    }

    const int assigned_partition = inner_id < this->node_partition_.size()
                                       ? this->node_partition_[inner_id]
                                       : kCspgUnassignedPartition;
    auto target_partitions = this->get_cspg_target_partitions(assigned_partition);

    for (int partition_id : target_partitions) {
        auto partition_graph = this->get_cspg_partition_graph(partition_id);
        if (partition_graph == nullptr) {
            continue;
        }

        InnerSearchParam partition_param;
        partition_param.topk = 1;
        partition_param.ef = 1;
        partition_param.is_inner_id_allowed = nullptr;
        // Paper-style CSPG build: each partition graph is built independently from its own
        // partition entry point, without descending partition route graphs to choose the seed.
        if (static_cast<size_t>(partition_id) < this->cspg_partition_entry_points_.size() &&
            this->cspg_partition_entry_points_[partition_id] != INVALID_ENTRY_POINT) {
            partition_param.ep = this->cspg_partition_entry_points_[partition_id];
        } else {
            partition_param.ep = inner_id;
        }

        partition_param.ef = this->ef_construct_;
        partition_param.topk = static_cast<int64_t>(this->ef_construct_);
        if (this->label_table_->CompressDuplicateData()) {
            partition_param.find_duplicate = true;
        }

        if (partition_graph->TotalCount() != 0) {
            result = search_one_graph(data,
                                      partition_graph,
                                      flatten_codes,
                                      partition_param,
                                      (VisitedListPtr) nullptr,
                                      nullptr);
            if (this->label_table_->CompressDuplicateData() && partition_param.duplicate_id >= 0) {
                std::unique_lock lock(this->label_lookup_mutex_);
                label_table_->SetDuplicateId(static_cast<InnerIdType>(partition_param.duplicate_id),
                                             inner_id);
                return false;
            }
            auto filtered_result = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
            while (not result->Empty()) {
                auto [dist, id] = result->Top();
                result->Pop();
                if (id != inner_id) {
                    filtered_result->Push(dist, id);
                }
            }
            LockGuard cur_lock(neighbors_mutex_, inner_id);
            mutually_connect_new_element(inner_id,
                                         filtered_result,
                                         partition_graph,
                                         flatten_codes,
                                         neighbors_mutex_,
                                         allocator_,
                                         alpha_);
            if (static_cast<size_t>(partition_id) < this->cspg_partition_entry_points_.size() &&
                (partition_graph->TotalCount() == 1 ||
                 partition_graph->GetNeighborSize(inner_id) > 0)) {
                auto& entry_point = this->cspg_partition_entry_points_[partition_id];
                if (entry_point == INVALID_ENTRY_POINT ||
                    !partition_graph->CheckIdExists(entry_point)) {
                    entry_point = inner_id;
                }
            }
        } else {
            LockGuard cur_lock(neighbors_mutex_, inner_id);
            partition_graph->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
            if (static_cast<size_t>(partition_id) < this->cspg_partition_entry_points_.size()) {
                auto& entry_point = this->cspg_partition_entry_points_[partition_id];
                if (entry_point == INVALID_ENTRY_POINT ||
                    !partition_graph->CheckIdExists(entry_point)) {
                    entry_point = inner_id;
                }
            }
        }

    }
    return true;
}

void
HGraph::resize(uint64_t new_size) {
    auto cur_size = this->max_capacity_.load();
    uint64_t new_size_power_2 =
        next_multiple_of_power_of_two(new_size, this->resize_increase_count_bit_);
    if (cur_size >= new_size_power_2) {
        return;
    }
    std::scoped_lock lock(this->global_mutex_);
    cur_size = this->max_capacity_.load();
    if (cur_size < new_size_power_2) {
        this->neighbors_mutex_->Resize(new_size_power_2);
        pool_ = std::make_shared<VisitedListPool>(1, allocator_, new_size_power_2, allocator_);
        this->label_table_->Resize(new_size_power_2);
        bottom_graph_->Resize(new_size_power_2);

        if (not this->cspg_partition_graphs_.empty()) {
            for (auto& graph : this->cspg_partition_graphs_) {
                if (graph != nullptr) {
                    graph->Resize(new_size_power_2);
                }
            }
        }
        for (auto& route_graphs : this->cspg_partition_route_graphs_) {
            for (auto& graph : route_graphs) {
                if (graph != nullptr) {
                    graph->Resize(new_size_power_2);
                }
            }
        }

        // CSPG：同步扩容分区数组，保持与图容量一致。
        this->node_partition_.resize(new_size_power_2, kCspgUnassignedPartition);
        this->staged_node_partition_.resize(new_size_power_2, kCspgUnassignedPartition);

        this->basic_flatten_codes_->Resize(new_size_power_2);
        if (use_reorder_) {
            this->high_precise_codes_->Resize(new_size_power_2);
        }
        if (create_new_raw_vector_) {
            this->raw_vector_->Resize(new_size_power_2);
        }
        if (this->extra_infos_ != nullptr) {
            this->extra_infos_->Resize(new_size_power_2);
        }
        this->max_capacity_.store(new_size_power_2);
        this->cal_memory_usage();
    }
}
void
HGraph::InitFeatures() {
    // Common Init
    // Build & Add
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_BUILD_WITH_MULTI_THREAD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
        IndexFeature::SUPPORT_MERGE_INDEX,
    });
    // search
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
        IndexFeature::SUPPORT_KNN_ITERATOR_FILTER_SEARCH,
    });
    // update
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        this->index_feature_list_->SetFeatures({IndexFeature::SUPPORT_UPDATE_VECTOR_CONCURRENT});
    }
    this->index_feature_list_->SetFeatures({IndexFeature::SUPPORT_UPDATE_ID_CONCURRENT});
    // concurrency
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_SEARCH_CONCURRENT);
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_ADD_CONCURRENT);
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_ADD_SEARCH_CONCURRENT);
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_ADD_SEARCH_DELETE_CONCURRENT);
    // serialize
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_DESERIALIZE_FILE,
        IndexFeature::SUPPORT_DESERIALIZE_READER_SET,
        IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_SERIALIZE_FILE,
        IndexFeature::SUPPORT_SERIALIZE_WRITE_FUNC,
    });
    // other
    this->index_feature_list_->SetFeatures({IndexFeature::SUPPORT_ESTIMATE_MEMORY,
                                            IndexFeature::SUPPORT_GET_MEMORY_USAGE,
                                            IndexFeature::SUPPORT_CHECK_ID_EXIST,
                                            IndexFeature::SUPPORT_CLONE,
                                            IndexFeature::SUPPORT_EXPORT_MODEL,
                                            IndexFeature::SUPPORT_TUNE});

    // About Train
    auto name = this->basic_flatten_codes_->GetQuantizerName();

    if (name != QUANTIZATION_TYPE_VALUE_FP32 and name != QUANTIZATION_TYPE_VALUE_BF16) {
        this->index_feature_list_->SetFeature(IndexFeature::NEED_TRAIN);
    } else {
        this->index_feature_list_->SetFeatures({
            IndexFeature::SUPPORT_RANGE_SEARCH,
            IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
        });
    }

    bool have_fp32 = false;
    bool hold_molds = false;
    if (name == QUANTIZATION_TYPE_VALUE_FP32) {
        have_fp32 = true;
        hold_molds |= this->basic_flatten_codes_->HoldMolds();
    }
    if (use_reorder_ and not ignore_reorder_ and
        this->high_precise_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        have_fp32 = true;
        hold_molds |= this->high_precise_codes_->HoldMolds();
    }
    if (have_fp32) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID);
        if (metric_ != MetricType::METRIC_TYPE_COSINE || hold_molds) {
            this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS);
        }
    }

    if (raw_vector_ != nullptr) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS);
    }

    // metric
    if (metric_ == MetricType::METRIC_TYPE_IP) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_INNER_PRODUCT);
    } else if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_L2);
    } else if (metric_ == MetricType::METRIC_TYPE_COSINE) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_COSINE);
    }

    if (this->extra_infos_ != nullptr) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_EXTRA_INFO_BY_ID);
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_KNN_SEARCH_WITH_EX_FILTER);
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_UPDATE_EXTRA_INFO_CONCURRENT);
    }
}

void
HGraph::elp_optimize() {
    InnerSearchParam param;
    param.ep = 0;
    param.ef = 80;
    param.topk = 10;
    param.is_inner_id_allowed = nullptr;
    searcher_->SetMockParameters(bottom_graph_, basic_flatten_codes_, pool_, param, dim_);
    // TODO(ZXY): optimize PREFETCH_DEPTH_CODE and add default value for the others
    optimizer_->RegisterParameter(RuntimeParameter(PREFETCH_STRIDE_CODE, 1, 10, 1));
    optimizer_->RegisterParameter(RuntimeParameter(PREFETCH_STRIDE_VISIT, 1, 10, 1));
    optimizer_->Optimize(searcher_);
}

void
HGraph::reorder(const void* query,
                const FlattenInterfacePtr& flatten,
                DistHeapPtr& candidate_heap,
                int64_t k,
                IteratorFilterContext* iter_ctx,
                QueryContext& ctx) const {
    uint64_t size = candidate_heap->Size();
    if (k <= 0) {
        k = static_cast<int64_t>(size);
    }
    auto reorder_heap =
        reorder_->Reorder(candidate_heap, static_cast<const float*>(query), k, ctx, iter_ctx);
    candidate_heap = reorder_heap;
}

ParamPtr
HGraph::CheckAndMappingExternalParam(const JsonType& external_param,
                                     const IndexCommonParam& common_param) {
    auto inner_json = map_hgraph_param(external_param);
    if (common_param.data_type_ == DataTypes::DATA_TYPE_SPARSE) {
        inner_json[BASE_CODES_KEY][CODES_TYPE_KEY].SetString(SPARSE_CODES);
        inner_json[PRECISE_CODES_KEY][CODES_TYPE_KEY].SetString(SPARSE_CODES);
        inner_json[RAW_VECTOR_KEY][CODES_TYPE_KEY].SetString(SPARSE_CODES);
    }

    auto hgraph_parameter = std::make_shared<HGraphParameter>();
    hgraph_parameter->data_type = common_param.data_type_;
    hgraph_parameter->FromJson(inner_json);
    uint64_t max_degree = hgraph_parameter->bottom_graph_param->max_degree_;

    auto max_degree_threshold = std::max<int64_t>(common_param.dim_, 128);
    CHECK_ARGUMENT(  // NOLINT
        (4 <= max_degree) and (max_degree <= max_degree_threshold),
        fmt::format("max_degree({}) must in range[4, {}]", max_degree, max_degree_threshold));

    auto construction_threshold = std::max<uint64_t>(1000UL, AMPLIFICATION_FACTOR * max_degree);
    CHECK_ARGUMENT((max_degree <= hgraph_parameter->ef_construction) and  // NOLINT
                       (hgraph_parameter->ef_construction <= construction_threshold),
                   fmt::format("ef_construction({}) must in range[$max_degree({}), {}]",
                               hgraph_parameter->ef_construction,
                               max_degree,
                               construction_threshold));
    return hgraph_parameter;
}
InnerIndexPtr
HGraph::ExportModel(const IndexCommonParam& param) const {
    auto index = std::make_shared<HGraph>(this->create_param_ptr_, param);
    this->basic_flatten_codes_->ExportModel(index->basic_flatten_codes_);
    if (use_reorder_) {
        this->high_precise_codes_->ExportModel(index->high_precise_codes_);
    }
    return index;
}
void
HGraph::GetCodeByInnerId(InnerIdType inner_id, uint8_t* data) const {
    if (raw_vector_ != nullptr) {
        raw_vector_->GetCodesById(inner_id, data);
        return;
    }

    if (use_reorder_) {
        high_precise_codes_->GetCodesById(inner_id, data);
    } else {
        basic_flatten_codes_->GetCodesById(inner_id, data);
    }
}

uint32_t
HGraph::Remove(const std::vector<int64_t>& ids, RemoveMode mode) {
    uint32_t delete_count = 0;
    if (mode == RemoveMode::MARK_REMOVE) {
        std::scoped_lock label_lock(this->label_lookup_mutex_);
        delete_count = this->label_table_->MarkRemove(ids);
        delete_count_ += delete_count;
        return delete_count;
    }
    for (const auto& id : ids) {
        InnerIdType inner_id;
        {
            std::shared_lock lock(this->label_lookup_mutex_);
            inner_id = this->label_table_->GetIdByLabel(id);
        }
        if (!this->is_cspg_enabled() && inner_id == this->entry_point_id_) {
            bool find_new_ep = false;
            while (not route_graphs_.empty()) {
                auto& upper_graph = route_graphs_.back();
                Vector<InnerIdType> neighbors(allocator_);
                upper_graph->GetNeighbors(this->entry_point_id_, neighbors);
                for (const auto& nb_id : neighbors) {
                    if (inner_id == nb_id) {
                        continue;
                    }
                    this->entry_point_id_ = nb_id;
                    find_new_ep = true;
                    break;
                }
                if (find_new_ep) {
                    break;
                }
                route_graphs_.pop_back();
            }
        }
        {
            {
                std::scoped_lock<std::shared_mutex> wlock(this->global_mutex_);
                for (int level = static_cast<int>(route_graphs_.size()) - 1; level >= 0; --level) {
                    this->route_graphs_[level]->DeleteNeighborsById(inner_id);
                }
                this->bottom_graph_->DeleteNeighborsById(inner_id);
                if (this->is_cspg_enabled() && !this->cspg_partition_graphs_.empty()) {
                    if (inner_id < this->node_partition_.size()) {
                        const int partition = this->node_partition_[inner_id];
                        for (auto target_partition : this->get_cspg_target_partitions(partition)) {
                            if (target_partition < 0 || static_cast<size_t>(target_partition) >=
                                                            this->cspg_partition_graphs_.size()) {
                                continue;
                            }
                            auto& graph =
                                this->cspg_partition_graphs_[static_cast<size_t>(target_partition)];
                            if (graph != nullptr) {
                                graph->DeleteNeighborsById(inner_id);
                            }
                            if (static_cast<size_t>(target_partition) <
                                this->cspg_partition_route_graphs_.size()) {
                                for (auto& route_graph :
                                     this->cspg_partition_route_graphs_[static_cast<size_t>(
                                         target_partition)]) {
                                    if (route_graph != nullptr) {
                                        route_graph->DeleteNeighborsById(inner_id);
                                    }
                                }
                            }
                        }
                    }
                }
                if (inner_id < this->node_partition_.size()) {
                    this->node_partition_[inner_id] = kCspgUnassignedPartition;
                }
                if (inner_id < this->staged_node_partition_.size()) {
                    this->staged_node_partition_[inner_id] = kCspgUnassignedPartition;
                }
            }
            std::scoped_lock label_lock(this->label_lookup_mutex_);
            this->label_table_->MarkRemove(id);
            delete_count++;
        }
        if (this->is_cspg_enabled()) {
            this->cspg_partition_entry_points_ =
                BuildCspgPartitionEntryPoints(this->node_partition_, this->cspg_m_);
            this->rebuild_cspg_partition_route_entries();
            if (inner_id == this->entry_point_id_) {
                this->entry_point_id_ = this->choose_any_cspg_entry_point();
            }
        }
    }
    return delete_count;
}

void
HGraph::ShrinkAndRepair(double timeout_ms) {
    HGraphShrinkContext ctx(this);
    ctx.Run(timeout_ms);
}

void
HGraph::recover_remove(int64_t id) {
    // note:
    // 1. this function doesn't recover entry_point and route_graphs caused by Remove()
    // 2. use this function only when is_tombstone is checked

    std::shared_lock label_lock(this->label_lookup_mutex_);
    auto inner_id = this->label_table_->GetIdByLabel(id, true);
    this->bottom_graph_->RecoverDeleteNeighborsById(inner_id);
    this->label_table_->RecoverRemove(id);
    delete_count_--;
}

DatasetPtr
HGraph::get_single_dataset(const DatasetPtr& data, uint32_t j) {
    void* vectors = nullptr;
    uint64_t data_size = 0;
    get_vectors(data_type_, dim_, data, &vectors, &data_size);
    const auto* labels = data->GetIds();
    auto one_data = Dataset::Make();
    one_data->Ids(labels + j)
        ->Float32Vectors((float*)((char*)vectors + data_size * j))
        ->Int8Vectors((int8_t*)((char*)vectors + data_size * j))
        ->NumElements(1)
        ->Owner(false);
    return one_data;
}

bool
HGraph::try_recover_tombstone(const DatasetPtr& data, std::vector<int64_t>& failed_ids) {
    /*
     * return:
     *      True : No processing required — data already exists or was recovered successfully
     *      False: Processing required — data not found or recovery failed
     *
     *
     * [case 1] fail to insert -> continue + record failed id
     * exist + not delete : is_label_valid = true, is_tombstone = false
     *
     * [case 2] fail to recovery -> add process
     * exist + delete + not recovery: is_label_valid = false, is_tombstone = ture, is_recovered = false
     *
     * [case 3] tombstone recovery -> continue
     * exist + delete + recovery: is_label_valid = false, is_tombstone = ture, is_recovered = true
     *
     * [case 4] no old point -> add process
     * not exists + not delete: is_label_valid = false, is_tombstone = false
     *
     * [case 5] error
     * exists + deleted: is_label_valid = true, is_tombstone = true
     */

    auto label = data->GetIds()[0];

    bool is_label_valid = false;
    bool is_tombstone = false;
    bool is_recovered = false;
    {
        std::scoped_lock label_lock(this->label_lookup_mutex_);
        is_label_valid = this->label_table_->CheckLabel(label);
        if (not is_label_valid) {
            is_tombstone = this->label_table_->IsTombstoneLabel(label);
        }
    }

    if (is_tombstone) {
        try {
            // try recover and update
            recover_remove(label);
            auto update_res = UpdateVector(label, data, false);
            if (update_res) {
                // [case 3]
                is_recovered = true;
                return is_recovered;
            }
            // recover failed: roll back
            Remove({label});
        } catch (std::runtime_error& e) {
            // recover failed: roll back
            Remove({label});
        }
    }

    // is_recovered = false
    if (is_label_valid) {
        // [case 1]
        failed_ids.emplace_back(label);
        return true;
    }

    // [case 2, 4]
    return false;
}

void
HGraph::Merge(const std::vector<MergeUnit>& merge_units) {
    int64_t total_count = this->GetNumElements();
    for (const auto& unit : merge_units) {
        total_count += unit.index->GetNumElements();
    }
    if (max_capacity_ < total_count) {
        this->resize(total_count);
    }
    for (const auto& merge_unit : merge_units) {
        const auto other_index = std::dynamic_pointer_cast<HGraph>(
            std::dynamic_pointer_cast<IndexImpl<HGraph>>(merge_unit.index)->GetInnerIndex());
        CHECK_ARGUMENT(this->cspg_m_ == other_index->cspg_m_ &&
                           std::abs(this->cspg_lambda_ - other_index->cspg_lambda_) <= 1e-6F,
                       "cspg parameters must be the same when merging HGraph indexes");
        CHECK_ARGUMENT(
            this->cspg_partition_graphs_.empty() == other_index->cspg_partition_graphs_.empty(),
            "cspg partition graph availability must match when merging HGraph indexes");
        auto merge_bias = this->total_count_.load();
        if (total_count_ == 0) {
            this->entry_point_id_ = other_index->entry_point_id_;
        }
        basic_flatten_codes_->MergeOther(other_index->basic_flatten_codes_, merge_bias);
        label_table_->MergeOther(other_index->label_table_, merge_unit.id_map_func);
        if (use_reorder_) {
            high_precise_codes_->MergeOther(other_index->high_precise_codes_, merge_bias);
        }
        bottom_graph_->MergeOther(other_index->bottom_graph_, merge_bias);
        while (route_graphs_.size() < other_index->route_graphs_.size()) {
            route_graphs_.push_back(this->generate_one_route_graph());
        }
        for (int j = 0; j < std::min(other_index->route_graphs_.size(), route_graphs_.size());
             ++j) {
            route_graphs_[j]->MergeOther(other_index->route_graphs_[j], merge_bias);
        }

        if (this->is_cspg_enabled() && !other_index->cspg_partition_graphs_.empty()) {
            this->ensure_cspg_partition_graphs();
            while (this->cspg_partition_graphs_.size() <
                   other_index->cspg_partition_graphs_.size()) {
                this->cspg_partition_graphs_.push_back(this->generate_one_partition_graph());
            }
            for (int j = 0; j < std::min(other_index->cspg_partition_graphs_.size(),
                                         this->cspg_partition_graphs_.size());
                 ++j) {
                this->cspg_partition_graphs_[j]->MergeOther(other_index->cspg_partition_graphs_[j],
                                                            merge_bias);
            }
            for (InnerIdType i = 0; i < other_index->GetNumElements(); ++i) {
                InnerIdType target_id = merge_bias + i;
                if (target_id >= this->node_partition_.size()) {
                    break;
                }
                if (i < other_index->node_partition_.size()) {
                    this->node_partition_[target_id] = other_index->node_partition_[i];
                } else {
                    this->node_partition_[target_id] = kCspgUnassignedPartition;
                }
            }
        }
        this->total_count_ += other_index->GetNumElements();
    }
    if (this->odescent_param_ == nullptr) {
        odescent_param_ = std::make_shared<ODescentParameter>();
    }

    auto build_data = (use_reorder_ and not build_by_base_) ? this->high_precise_codes_
                                                            : this->basic_flatten_codes_;
    for (InnerIdType inner_id = 0; inner_id < this->total_count_; ++inner_id) {
        Vector<InnerIdType> neighbors(this->allocator_);
        this->bottom_graph_->GetNeighbors(inner_id, neighbors);
        neighbors.resize(neighbors.size() / 2);
        this->bottom_graph_->InsertNeighborsById(inner_id, neighbors);
    }
    {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree();
        ODescent odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        odescent_builder.Build(bottom_graph_);
        odescent_builder.SaveGraph(bottom_graph_);
    }
    for (auto& graph : route_graphs_) {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree() / 2;
        ODescent sparse_odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        auto ids = graph->GetIds();
        sparse_odescent_builder.Build(ids, graph);
        sparse_odescent_builder.SaveGraph(graph);
        this->entry_point_id_ = ids.back();
    }
    if (this->is_cspg_enabled()) {
        this->cspg_partition_entry_points_ =
            BuildCspgPartitionEntryPoints(this->node_partition_, this->cspg_m_);
        this->staged_node_partition_.assign(this->max_capacity_.load(), kCspgUnassignedPartition);
    }
}

void
HGraph::GetVectorByInnerId(InnerIdType inner_id, float* data) const {
    auto codes = (use_reorder_) ? high_precise_codes_ : basic_flatten_codes_;
    codes = (create_new_raw_vector_) ? raw_vector_ : codes;
    bool release;
    const auto* buffer = codes->GetCodesById(inner_id, release);
    codes->Decode(buffer, data);
    if (release) {
        codes->Release(buffer);
    }
}

void
HGraph::SetImmutable() {
    if (this->immutable_) {
        return;
    }
    std::scoped_lock<std::shared_mutex> wlock(this->global_mutex_);
    this->neighbors_mutex_.reset();
    this->neighbors_mutex_ = std::make_shared<EmptyMutex>();
    this->searcher_->SetMutexArray(this->neighbors_mutex_);
    this->immutable_ = true;
}

void
HGraph::SetIO(const std::shared_ptr<Reader> reader) {
    auto reader_param = std::make_shared<ReaderIOParameter>();
    reader_param->reader = reader;
    if (use_reorder_) {
        high_precise_codes_->InitIO(reader_param);
    }
    basic_flatten_codes_->InitIO(reader_param);
    bottom_graph_->InitIO(reader_param);
}

[[nodiscard]] DatasetPtr
HGraph::SearchWithRequest(const SearchRequest& request) const {
    SearchStatistics stats;
    QueryContext ctx{.alloc = this->allocator_, .stats = &stats};
    if (request.search_allocator_ != nullptr) {
        ctx.alloc = request.search_allocator_;
    }

    const auto& query = request.query_;
    int64_t query_dim = query->GetDim();
    auto k = request.topk_;
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    }

    auto params = HGraphSearchParameters::FromJson(request.params_str_);
    stats.enable_cspg_stats.store(params.cspg_enable_stats, std::memory_order_relaxed);

    auto ef_search_threshold = std::max<int64_t>(AMPLIFICATION_FACTOR * k, 1000);
    CHECK_ARGUMENT(  // NOLINT
        (1 <= params.ef_search) and (params.ef_search <= ef_search_threshold),
        fmt::format("ef_search({}) must in range[1, {}]", params.ef_search, ef_search_threshold));

    std::shared_lock shared_lock(this->global_mutex_);
    // check k
    CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k));
    k = std::min(k, GetNumElements());

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    InnerSearchParam search_param;
    search_param.ep = this->entry_point_id_;
    search_param.topk = 1;
    search_param.ef = 1;
    search_param.is_inner_id_allowed = nullptr;

    if (search_param.ep == INVALID_ENTRY_POINT) {
        SearchStatistics stats;
        auto dataset_result = DatasetImpl::MakeEmptyDataset();
        dataset_result->Statistics(stats.Dump());
        return dataset_result;
    }

    auto vt = this->pool_->TakeOne();

    const auto* raw_query = get_data(query);
    const bool has_cspg_partition_graph_data =
        std::any_of(this->cspg_partition_graphs_.begin(),
                    this->cspg_partition_graphs_.end(),
                    [](const GraphInterfacePtr& graph) {
                        return graph != nullptr && graph->TotalCount() != 0;
                    });
    bool enable_cspg = this->is_cspg_enabled() && has_cspg_partition_graph_data;
    if (!enable_cspg) {
        for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
            auto result = this->search_one_graph(raw_query,
                                                 this->route_graphs_[i],
                                                 this->basic_flatten_codes_,
                                                 search_param,
                                                 vt,
                                                 &ctx);
            search_param.ep = result->Top().second;
        }
    }

    auto combined_filter = std::make_shared<CombinedFilter>();
    combined_filter->AppendFilter(this->label_table_->GetDeletedIdsFilter());
    if (request.filter_ != nullptr) {
        if (params.use_extra_info_filter) {
            combined_filter->AppendFilter(
                std::make_shared<ExtraInfoWrapperFilter>(request.filter_, this->extra_infos_));
        } else {
            combined_filter->AppendFilter(
                std::make_shared<InnerIdWrapperFilter>(request.filter_, *this->label_table_));
        }
    }
    FilterPtr ft = nullptr;
    if (not combined_filter->IsEmpty()) {
        ft = combined_filter;
    }

    if (request.enable_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        auto& schema = this->attr_filter_index_->field_type_map_;
        auto expr = AstParse(request.attribute_filter_str_, &schema);
        auto executor = Executor::MakeInstance(this->allocator_, expr, this->attr_filter_index_);
        executor->Init();
        search_param.executors.emplace_back(executor);
    }

    search_param.ef = std::max(params.ef_search, k);
    search_param.is_inner_id_allowed = ft;
    search_param.topk = static_cast<int64_t>(search_param.ef);
    if (params.topk_factor > 1.0F) {
        search_param.topk = std::min(
            search_param.topk, static_cast<int64_t>(static_cast<float>(k) * params.topk_factor));
    }
    search_param.consider_duplicate = true;
    if (params.enable_time_record) {
        search_param.time_cost = std::make_shared<Timer>();
        search_param.time_cost->SetThreshold(params.timeout_ms);
        stats.is_timeout.store(false, std::memory_order_relaxed);
    }
    search_param.parallel_search_thread_count = params.parallel_search_thread_count;

    // hops_limit only takes effect when it's greater than ef_search
    if (params.hops_limit <= static_cast<uint32_t>(params.ef_search)) {
        search_param.hops_limit = std::numeric_limits<uint32_t>::max();
        if (params.hops_limit != std::numeric_limits<uint32_t>::max()) {
            logger::warn(
                fmt::format("hops_limit({}) is not greater than ef_search({}), ignoring hops_limit",
                            params.hops_limit,
                            params.ef_search));
        }
    } else {
        search_param.hops_limit = params.hops_limit;
    }

    auto search_result = DistHeapPtr{};

    auto choose_entry_point = [&](const GraphInterfacePtr& graph,
                                  InnerIdType preferred_ep,
                                  size_t partition_id) -> InnerIdType {
        if (graph == nullptr || graph->TotalCount() == 0) {
            return INVALID_ENTRY_POINT;
        }
        auto is_valid_entry = [&](InnerIdType id) -> bool {
            return id != INVALID_ENTRY_POINT && graph->CheckIdExists(id);
        };

        if (is_valid_entry(preferred_ep)) {
            return preferred_ep;
        }
        if (partition_id < this->cspg_partition_entry_points_.size()) {
            auto entry = this->cspg_partition_entry_points_[partition_id];
            if (is_valid_entry(entry)) {
                return entry;
            }
        }
        return INVALID_ENTRY_POINT;
    };

    if (enable_cspg) {
        // CSPG query is split into two stages:
        // 1) fast approaching within one partition;
        // 2) cross-partition expansion with a larger candidate set.
        int64_t ef1 = std::max<int64_t>(1, params.cspg_ef1);
        const uint64_t ef2 = params.cspg_ef2 > 0
                                 ? static_cast<uint64_t>(std::max<int64_t>(params.cspg_ef2, k))
                                 : search_param.ef;

        struct CspgPhase1Seed {
            float dist;
            InnerIdType id;
            size_t partition_id;
        };
        std::vector<CspgPhase1Seed> phase1_seeds;
        uint32_t hops = 0;
        uint32_t dist_cmp = 0;
        auto computer = this->basic_flatten_codes_->FactoryComputer(raw_query);

        std::vector<size_t> phase1_partitions;
        phase1_partitions.reserve(this->cspg_partition_graphs_.size());
        for (size_t partition_id = 0; partition_id < this->cspg_partition_graphs_.size();
             ++partition_id) {
            auto partition_graph = this->get_cspg_partition_graph(static_cast<int>(partition_id));
            if (partition_graph != nullptr && partition_graph->TotalCount() != 0) {
                phase1_partitions.emplace_back(partition_id);
            }
        }
        auto run_phase1_on_partition = [&](size_t partition_id) -> std::pair<InnerIdType, float> {
            auto partition_graph = this->get_cspg_partition_graph(static_cast<int>(partition_id));
            if (partition_graph == nullptr || partition_graph->TotalCount() == 0) {
                return {INVALID_ENTRY_POINT, std::numeric_limits<float>::max()};
            }

            InnerIdType phase1_preferred_ep = INVALID_ENTRY_POINT;
            uint32_t route_phase1_hops = 0;
            uint32_t route_phase1_dist_cmp = 0;
            if (params.cspg_phase1_use_route_descent) {
                int route_entry_level = -1;
                InnerIdType route_entry = INVALID_ENTRY_POINT;
                // 这是 HGraph 风格增强路径，不属于论文 Algorithm 1 的默认流程。
                const auto has_cached_route_entry =
                    partition_id < this->cspg_partition_route_entry_points_.size() &&
                    partition_id < this->cspg_partition_route_entry_levels_.size();
                if (has_cached_route_entry) {
                    route_entry = this->cspg_partition_route_entry_points_[partition_id];
                    route_entry_level = this->cspg_partition_route_entry_levels_[partition_id];
                    const auto has_route_graphs =
                        partition_id < this->cspg_partition_route_graphs_.size();
                    const auto cached_level_valid =
                        route_entry != INVALID_ENTRY_POINT && route_entry_level >= 0 &&
                        has_route_graphs &&
                        static_cast<size_t>(route_entry_level) <
                            this->cspg_partition_route_graphs_[partition_id].size();
                    if (!cached_level_valid) {
                        route_entry = INVALID_ENTRY_POINT;
                        route_entry_level = -1;
                    } else {
                        const auto& cached_route_graph =
                            this->cspg_partition_route_graphs_[partition_id]
                                                              [static_cast<size_t>(
                                                                  route_entry_level)];
                        if (cached_route_graph == nullptr ||
                            !cached_route_graph->CheckIdExists(route_entry)) {
                            route_entry = INVALID_ENTRY_POINT;
                            route_entry_level = -1;
                        }
                    }
                }
                if (route_entry == INVALID_ENTRY_POINT) {
                    route_entry = this->find_cspg_partition_route_entry(
                        static_cast<int>(partition_id), &route_entry_level);
                }
                if (route_entry != INVALID_ENTRY_POINT) {
                    // 可选增强：先沿 partition route graph 自顶向下下降，再进入 phase-1。
                    InnerSearchParam route_search_param;
                    route_search_param.ep = route_entry;
                    route_search_param.topk = 1;
                    route_search_param.ef = 1;
                    route_search_param.is_inner_id_allowed = nullptr;
                    route_search_param.hops_limit = search_param.hops_limit;
                    route_search_param.time_cost = search_param.time_cost;
                    route_search_param.parallel_search_thread_count =
                        search_param.parallel_search_thread_count;
                    phase1_preferred_ep = route_entry;
                    const auto* route_graphs =
                        partition_id < this->cspg_partition_route_graphs_.size()
                            ? &this->cspg_partition_route_graphs_[partition_id]
                            : nullptr;
                    for (int current_level = route_entry_level; current_level >= 0;
                         --current_level) {
                        if (route_graphs != nullptr &&
                            static_cast<size_t>(current_level) < route_graphs->size()) {
                            auto route_graph = (*route_graphs)[static_cast<size_t>(current_level)];
                            if (route_graph == nullptr || route_graph->TotalCount() == 0) {
                                continue;
                            }
                            SearchStatistics route_stats;
                            QueryContext route_ctx{.alloc = ctx.alloc, .stats = &route_stats};
                            auto route_result = this->search_one_graph(raw_query,
                                                                       route_graph,
                                                                       this->basic_flatten_codes_,
                                                                       route_search_param,
                                                                       vt,
                                                                       &route_ctx);
                            route_phase1_hops +=
                                route_stats.hops.load(std::memory_order_relaxed);
                            route_phase1_dist_cmp +=
                                route_stats.dist_cmp.load(std::memory_order_relaxed);
                            if (route_stats.is_timeout.load(std::memory_order_relaxed)) {
                                stats.is_timeout.store(true, std::memory_order_relaxed);
                            }
                            if (route_result != nullptr && !route_result->Empty()) {
                                route_search_param.ep = route_result->Top().second;
                                phase1_preferred_ep = route_search_param.ep;
                            }
                        }
                    }
                }
            }

            const auto phase1_entry =
                choose_entry_point(partition_graph, phase1_preferred_ep, partition_id);
            if (phase1_entry == INVALID_ENTRY_POINT) {
                return {INVALID_ENTRY_POINT, std::numeric_limits<float>::max()};
            }

            struct Phase1State {
                float dist;
                InnerIdType id;
            };
            auto phase1_state_less = [](const Phase1State& lhs, const Phase1State& rhs) {
                if (lhs.dist != rhs.dist) {
                    return lhs.dist < rhs.dist;
                }
                return lhs.id < rhs.id;
            };
            auto phase1_state_worse_first = [](const Phase1State& lhs, const Phase1State& rhs) {
                if (lhs.dist != rhs.dist) {
                    return lhs.dist > rhs.dist;
                }
                return lhs.id > rhs.id;
            };

            vt->Reset();
            Vector<Phase1State> phase1_frontier(ctx.alloc);
            phase1_frontier.reserve(static_cast<size_t>(std::max<int64_t>(16, ef1)));
            auto enqueue_phase1_state = [&](InnerIdType id, float dist) {
                if (id == INVALID_ENTRY_POINT) {
                    return false;
                }
                const auto state = Phase1State{dist, id};
                if (!phase1_frontier.empty() &&
                    phase1_frontier.size() >= static_cast<size_t>(ef1) &&
                    !phase1_state_less(state, phase1_frontier.front())) {
                    return false;
                }
                const auto insert_it = std::lower_bound(phase1_frontier.begin(),
                                                        phase1_frontier.end(),
                                                        state,
                                                        phase1_state_worse_first);
                phase1_frontier.insert(insert_it, state);
                if (phase1_frontier.size() > static_cast<size_t>(ef1)) {
                    phase1_frontier.erase(phase1_frontier.begin());
                }
                return true;
            };

            uint32_t phase1_hops = route_phase1_hops;
            uint32_t phase1_dist_cmp = route_phase1_dist_cmp;
            float best_phase1_dist = std::numeric_limits<float>::max();
            InnerIdType best_phase1_id = INVALID_ENTRY_POINT;
            auto phase1_results =
                DistanceHeap::MakeInstanceBySize<true, true>(ctx.alloc, static_cast<int64_t>(ef1));
            this->basic_flatten_codes_->Query(&best_phase1_dist, computer, &phase1_entry, 1, &ctx);
            ++phase1_dist_cmp;
            best_phase1_id = phase1_entry;
            vt->Set(phase1_entry);
            phase1_results->Push(best_phase1_dist, phase1_entry);
            enqueue_phase1_state(phase1_entry, best_phase1_dist);
            auto phase1_bound = [&]() {
                if (phase1_results->Size() < static_cast<uint64_t>(ef1) ||
                    phase1_results->Empty()) {
                    return std::numeric_limits<float>::max();
                }
                return phase1_results->Top().first;
            };

            Vector<InnerIdType> phase1_neighbors(ctx.alloc);
            Vector<InnerIdType> phase1_candidate_neighbors(ctx.alloc);
            Vector<float> phase1_dists(ctx.alloc);
            const auto phase1_capacity = partition_graph->MaximumDegree();
            phase1_neighbors.reserve(phase1_capacity);
            phase1_candidate_neighbors.resize(phase1_capacity);
            phase1_dists.resize(phase1_capacity);

            while (!phase1_frontier.empty()) {
                if (search_param.hops_limit != std::numeric_limits<uint32_t>::max() &&
                    phase1_hops >= search_param.hops_limit) {
                    break;
                }
                if (search_param.time_cost != nullptr && search_param.time_cost->CheckOvertime()) {
                    stats.is_timeout.store(true, std::memory_order_relaxed);
                    break;
                }

                const auto current = phase1_frontier.back();
                if (phase1_results->Size() >= static_cast<uint64_t>(ef1) &&
                    current.dist > phase1_bound()) {
                    break;
                }
                phase1_frontier.pop_back();
                ++phase1_hops;

                phase1_neighbors.clear();
                partition_graph->GetNeighbors(current.id, phase1_neighbors);
                if (phase1_neighbors.empty()) {
                    continue;
                }

                uint32_t candidate_count = 0;
                for (size_t i = 0; i < phase1_neighbors.size(); ++i) {
                    const auto neighbor = phase1_neighbors[i];
                    if (i + kCspgVisitPrefetchStride < phase1_neighbors.size()) {
                        vt->Prefetch(phase1_neighbors[i + kCspgVisitPrefetchStride]);
                    }
                    if (!vt->Get(neighbor)) {
                        vt->Set(neighbor);
                        phase1_candidate_neighbors[candidate_count++] = neighbor;
                    }
                }
                if (candidate_count == 0) {
                    continue;
                }

                this->basic_flatten_codes_->Query(phase1_dists.data(),
                                                  computer,
                                                  phase1_candidate_neighbors.data(),
                                                  candidate_count,
                                                  &ctx);
                phase1_dist_cmp += candidate_count;

                for (uint32_t i = 0; i < candidate_count; ++i) {
                    const auto neighbor = phase1_candidate_neighbors[i];
                    const auto neighbor_dist = phase1_dists[i];
                    if (neighbor_dist < best_phase1_dist ||
                        (neighbor_dist == best_phase1_dist && neighbor < best_phase1_id)) {
                        best_phase1_dist = neighbor_dist;
                        best_phase1_id = neighbor;
                    }
                    const bool phase1_full_before =
                        phase1_results->Size() >= static_cast<uint64_t>(ef1);
                    const auto bound_before_insert = phase1_bound();
                    if (!phase1_full_before || neighbor_dist < phase1_results->Top().first) {
                        phase1_results->Push(neighbor_dist, neighbor);
                    }
                    if (phase1_full_before && neighbor_dist >= bound_before_insert) {
                        continue;
                    }
                    enqueue_phase1_state(neighbor, neighbor_dist);
                }
            }

            if (ctx.stats != nullptr) {
                ctx.stats->hops.fetch_add(phase1_hops, std::memory_order_relaxed);
                ctx.stats->dist_cmp.fetch_add(phase1_dist_cmp, std::memory_order_relaxed);
                if (params.cspg_enable_stats) {
                    ctx.stats->cspg_phase1_hops.fetch_add(phase1_hops,
                                                          std::memory_order_relaxed);
                    ctx.stats->cspg_phase1_dist_cmp.fetch_add(phase1_dist_cmp,
                                                              std::memory_order_relaxed);
                }
            }
            return {best_phase1_id, best_phase1_dist};
        };

        // 默认保持论文风格：只在第一个非空 partition 上做 phase-1。
        // 当该 seed 质量不稳定时，允许按参数探测多个 partition，
        // 但 phase-2 仍只从最好的单个 seed 开始，避免初始 frontier 过宽。
        const auto phase1_partition_probe_count = std::min<size_t>(
            static_cast<size_t>(std::max<int64_t>(1, params.cspg_phase1_partition_count)),
            phase1_partitions.size());
        phase1_seeds.reserve(phase1_partition_probe_count);
        for (size_t i = 0; i < phase1_partition_probe_count; ++i) {
            const auto partition_id = phase1_partitions[i];
            vt->Reset();
            auto [candidate_seed, candidate_dist] = run_phase1_on_partition(partition_id);
            if (candidate_seed != INVALID_ENTRY_POINT) {
                phase1_seeds.push_back(
                    CspgPhase1Seed{candidate_dist, candidate_seed, partition_id});
            }
        }
        std::stable_sort(phase1_seeds.begin(),
                         phase1_seeds.end(),
                         [](const CspgPhase1Seed& lhs, const CspgPhase1Seed& rhs) {
                             if (lhs.dist != rhs.dist) {
                                 return lhs.dist < rhs.dist;
                             }
                             if (lhs.id != rhs.id) {
                                 return lhs.id < rhs.id;
                             }
                             return lhs.partition_id < rhs.partition_id;
                         });
        vt->Reset();

        struct CspgState {
            float dist;
            InnerIdType id;
            size_t partition_id;
            bool from_cross_partition;
            uint16_t cross_partition_depth;
            uint16_t cross_partition_switches;
        };
        struct CspgStateWorseFirstCompare {
            bool
            operator()(const CspgState& a, const CspgState& b) const {
                if (a.dist != b.dist) {
                    return a.dist > b.dist;
                }
                if (a.id != b.id) {
                    return a.id > b.id;
                }
                if (a.from_cross_partition != b.from_cross_partition) {
                    return a.from_cross_partition > b.from_cross_partition;
                }
                if (a.cross_partition_depth != b.cross_partition_depth) {
                    return a.cross_partition_depth > b.cross_partition_depth;
                }
                if (a.cross_partition_switches != b.cross_partition_switches) {
                    return a.cross_partition_switches > b.cross_partition_switches;
                }
                return a.partition_id > b.partition_id;
            }
        };
        // Active stage-2 states live in a binary min-heap keyed on distance, so
        // enqueue/pop are O(log n) instead of the O(n) shifts of a sorted vector.
        // The result-bound check gates what enters the frontier, so no separate
        // ef2 trim is needed.
        Vector<CspgState> frontier(ctx.alloc);
        const auto frontier_hint =
            static_cast<size_t>(std::max<uint64_t>(16, std::min<uint64_t>(ef2, 4096)));
        frontier.reserve(frontier_hint);
        auto stage2_results =
            DistanceHeap::MakeInstanceBySize<true, true>(ctx.alloc, static_cast<int64_t>(ef2));
        auto visited_topk_results =
            DistanceHeap::MakeInstanceBySize<true, true>(ctx.alloc, std::max<int64_t>(1, k));
        const bool collect_cspg_stats = ctx.stats != nullptr && params.cspg_enable_stats;
        uint32_t stage2_routing_pops = 0;
        uint32_t stage2_nonrouting_pops = 0;
        uint32_t stage2_routing_useless_pops = 0;
        uint32_t stage2_nonrouting_useless_pops = 0;
        uint32_t stage2_local_dist_cmp = 0;
        uint32_t stage2_cross_partition_dist_cmp = 0;
        uint32_t stage2_local_routing_dist_cmp = 0;
        uint32_t stage2_local_nonrouting_dist_cmp = 0;
        uint32_t stage2_routing_fanout_attempts = 0;
        uint32_t stage2_routing_fanout_enqueues = 0;
        uint32_t stage2_routing_fanout_skipped_unexpandable = 0;
        uint32_t stage2_routing_fanout_skipped_no_unvisited = 0;
        uint32_t stage2_routing_fanout_skipped_duplicate_state = 0;
        uint32_t stage2_routing_fanout_skipped_frontier_reject = 0;
        uint32_t stage2_routing_fanout_skipped_by_bound = 0;
        const auto partition_count = this->cspg_partition_graphs_.size();

        // std::*_heap with this comparator yields a heap whose front() is the
        // smallest-distance (best) state, since "worse-first" makes larger
        // distances compare as smaller.
        CspgStateWorseFirstCompare state_worse_first;
        auto current_bound = [&]() {
            if (stage2_results->Size() < ef2 || stage2_results->Empty()) {
                return std::numeric_limits<float>::max();
            }
            return stage2_results->Top().first;
        };
        auto enqueue_state = [&](InnerIdType id,
                                 size_t partition_id,
                                 float dist,
                                 bool from_cross_partition = false,
                                 uint16_t cross_partition_depth = 0,
                                 uint16_t cross_partition_switches = 0) -> bool {
            if (id == INVALID_ENTRY_POINT) {
                return false;
            }
            // Drop states no better than the ef2-th result: they could never
            // enter the result set, so exploring them is pure overhead.
            if (stage2_results->Size() >= ef2 && dist >= current_bound()) {
                return false;
            }
            frontier.push_back(CspgState{dist,
                                         id,
                                         partition_id,
                                         from_cross_partition,
                                         cross_partition_depth,
                                         cross_partition_switches});
            std::push_heap(frontier.begin(), frontier.end(), state_worse_first);
            return true;
        };

        auto consider_result = [&](InnerIdType id, float dist) -> bool {
            if (search_param.is_inner_id_allowed != nullptr &&
                !search_param.is_inner_id_allowed->CheckValid(id)) {
                return false;
            }
            visited_topk_results->Push(dist, id);
            if (stage2_results->Size() >= ef2 && dist >= stage2_results->Top().first) {
                return false;
            }
            stage2_results->Push(dist, id);
            return true;
        };

        auto is_cspg_routing_vector = [&](InnerIdType id) -> bool {
            return id >= 0 && static_cast<size_t>(id) < this->node_partition_.size() &&
                   this->node_partition_[static_cast<size_t>(id)] == kCspgRoutingPartition;
        };

        const auto partition_max_degree =
            !this->cspg_partition_graphs_.empty() && this->cspg_partition_graphs_[0] != nullptr
                ? this->cspg_partition_graphs_[0]->MaximumDegree()
                : this->bottom_graph_->MaximumDegree();
        Vector<InnerIdType> neighbors(ctx.alloc);
        Vector<InnerIdType> routing_neighbors(ctx.alloc);
        Vector<InnerIdType> candidate_neighbors(ctx.alloc);
        Vector<float> line_dists(ctx.alloc);
        neighbors.reserve(partition_max_degree);
        routing_neighbors.reserve(partition_max_degree);
        candidate_neighbors.resize(partition_max_degree);
        line_dists.resize(partition_max_degree);

        if (!phase1_seeds.empty()) {
            const auto& phase1_seed = phase1_seeds.front();
            enqueue_state(phase1_seed.id, phase1_seed.partition_id, phase1_seed.dist, false, 0, 0);
            if (!vt->Get(phase1_seed.id)) {
                vt->Set(phase1_seed.id);
                consider_result(phase1_seed.id, phase1_seed.dist);
            }
        }

        auto expand_one_partition = [&](const CspgState& current,
                                        bool current_is_routing,
                                        bool& current_pop_useful,
                                        float& result_bound) -> bool {
            const auto id = current.id;
            const auto partition_id = current.partition_id;
            if (partition_id >= this->cspg_partition_graphs_.size()) {
                return false;
            }
            const auto& partition_graph = this->cspg_partition_graphs_[partition_id];
            if (partition_graph == nullptr || partition_graph->TotalCount() == 0) {
                return false;
            }
            if (current.from_cross_partition && params.cspg_cross_partition_hops_limit > 0 &&
                current.cross_partition_depth >=
                    static_cast<uint16_t>(params.cspg_cross_partition_hops_limit)) {
                return false;
            }

            neighbors.clear();
            partition_graph->GetNeighbors(id, neighbors);
            if (neighbors.empty()) {
                return false;
            }

            // 先批量收集未访问邻居，再统一做距离计算，减少逐点 Query 的热路径开销。
            uint32_t candidate_count = 0;
            const bool limit_local_routing_neighbors = current.from_cross_partition &&
                                                       current_is_routing &&
                                                       params.cspg_local_routing_budget > 0;
            bool skipped_local_routing_neighbor = false;
            auto collect_neighbors = [&](bool allow_local_routing_neighbors) {
                for (size_t i = 0; i < neighbors.size(); ++i) {
                    auto neighbor = neighbors[i];
                    if (i + kCspgVisitPrefetchStride < neighbors.size()) {
                        vt->Prefetch(neighbors[i + kCspgVisitPrefetchStride]);
                    }
                    const bool neighbor_is_routing = is_cspg_routing_vector(neighbor);
                    if (limit_local_routing_neighbors && neighbor_is_routing &&
                        !allow_local_routing_neighbors) {
                        skipped_local_routing_neighbor = true;
                        continue;
                    }
                    if (!vt->Get(neighbor)) {
                        vt->Set(neighbor);
                        candidate_neighbors[candidate_count++] = neighbor;
                    }
                }
            };
            collect_neighbors(!limit_local_routing_neighbors);
            if (limit_local_routing_neighbors && skipped_local_routing_neighbor) {
                const auto candidate_capacity = static_cast<uint32_t>(candidate_neighbors.size());
                const uint32_t routing_budget = std::min<uint32_t>(
                    static_cast<uint32_t>(params.cspg_local_routing_budget),
                    candidate_capacity > candidate_count ? candidate_capacity - candidate_count
                                                         : 0);
                uint32_t added_local_routing_neighbors = 0;
                for (size_t i = 0;
                     i < neighbors.size() && added_local_routing_neighbors < routing_budget;
                     ++i) {
                    auto neighbor = neighbors[i];
                    if (!is_cspg_routing_vector(neighbor) || vt->Get(neighbor)) {
                        continue;
                    }
                    vt->Set(neighbor);
                    candidate_neighbors[candidate_count++] = neighbor;
                    ++added_local_routing_neighbors;
                }
            }
            if (candidate_count == 0) {
                return false;
            }

            this->basic_flatten_codes_->Query(
                line_dists.data(), computer, candidate_neighbors.data(), candidate_count, &ctx);
            dist_cmp += candidate_count;
            if (collect_cspg_stats) {
                if (current.from_cross_partition) {
                    stage2_cross_partition_dist_cmp += candidate_count;
                } else {
                    stage2_local_dist_cmp += candidate_count;
                    if (current_is_routing) {
                        stage2_local_routing_dist_cmp += candidate_count;
                    } else {
                        stage2_local_nonrouting_dist_cmp += candidate_count;
                    }
                }
            }

            auto enqueue_routing_instances = [&](InnerIdType routing_id, float routing_dist) {
                if (routing_id == INVALID_ENTRY_POINT) {
                    return false;
                }
                if (params.cspg_cross_partition_switch_limit > 0 &&
                    current.cross_partition_switches >=
                        static_cast<uint16_t>(params.cspg_cross_partition_switch_limit)) {
                    return false;
                }
                auto enqueue_one_routing_instance = [&](size_t other_partition) -> bool {
                    if (other_partition == partition_id) {
                        return false;
                    }
                    if (collect_cspg_stats) {
                        ++stage2_routing_fanout_attempts;
                    }
                    const auto& other_graph = this->cspg_partition_graphs_[other_partition];
                    if (other_graph == nullptr || other_graph->TotalCount() == 0) {
                        if (collect_cspg_stats) {
                            ++stage2_routing_fanout_skipped_unexpandable;
                        }
                        return false;
                    }
                    if (enqueue_state(routing_id,
                                      other_partition,
                                      routing_dist,
                                      true,
                                      0,
                                      static_cast<uint16_t>(current.cross_partition_switches +
                                                            1))) {
                        if (collect_cspg_stats) {
                            ++stage2_routing_fanout_enqueues;
                        }
                        return true;
                    } else if (collect_cspg_stats) {
                        ++stage2_routing_fanout_skipped_frontier_reject;
                    }
                    return false;
                };

                if (partition_count == 2 && partition_id < 2) {
                    return enqueue_one_routing_instance(1 - partition_id);
                }

                bool inserted_any_state = false;
                int64_t fanout_budget = (params.cspg_max_routing_fanout > 0)
                                            ? params.cspg_max_routing_fanout
                                            : static_cast<int64_t>(partition_count);
                for (size_t other_partition = 0;
                     other_partition < partition_count && fanout_budget > 0;
                     ++other_partition) {
                    if (enqueue_one_routing_instance(other_partition)) {
                        --fanout_budget;
                        inserted_any_state = true;
                    }
                }
                return inserted_any_state;
            };

            for (uint32_t i = 0; i < candidate_count; ++i) {
                auto neighbor = candidate_neighbors[i];
                auto dist = line_dists[i];
                const bool improved_results = consider_result(neighbor, dist);
                if (stage2_results->Size() >= ef2 && dist >= result_bound) {
                    if (collect_cspg_stats) {
                        current_pop_useful = current_pop_useful || improved_results;
                    }
                    continue;
                }
                const bool inserted_into_frontier =
                    enqueue_state(neighbor,
                                  partition_id,
                                  dist,
                                  current.from_cross_partition,
                                  current.from_cross_partition
                                      ? static_cast<uint16_t>(current.cross_partition_depth + 1)
                                      : 0,
                                  current.cross_partition_switches);
                if (collect_cspg_stats) {
                    current_pop_useful =
                        current_pop_useful || inserted_into_frontier || improved_results;
                }
                if (improved_results) {
                    result_bound = current_bound();
                }
                if (is_cspg_routing_vector(neighbor)) {
                    const bool inserted_routing_states = enqueue_routing_instances(neighbor, dist);
                    if (collect_cspg_stats) {
                        current_pop_useful = current_pop_useful || inserted_routing_states;
                    }
                }
            }
            return true;
        };

        // Stage-2 follows Algorithm 1 as a greedy beam search: expand the closest
        // active state, keep L bounded by ef2, and cross partitions only when a
        // routing vector is encountered. Final top-k still comes from visited.
        while (!frontier.empty()) {
            if (params.hops_limit != std::numeric_limits<uint32_t>::max() &&
                hops >= params.hops_limit) {
                break;
            }
            if (search_param.time_cost != nullptr && search_param.time_cost->CheckOvertime()) {
                stats.is_timeout.store(true, std::memory_order_relaxed);
                break;
            }

            if (stage2_results->Size() >= ef2 && frontier.front().dist > current_bound()) {
                break;
            }
            std::pop_heap(frontier.begin(), frontier.end(), state_worse_first);
            const auto current = frontier.back();
            frontier.pop_back();
            ++hops;
            const bool current_is_routing = is_cspg_routing_vector(current.id);
            if (collect_cspg_stats) {
                if (current_is_routing) {
                    ++stage2_routing_pops;
                } else {
                    ++stage2_nonrouting_pops;
                }
            }
            bool current_pop_useful = false;
            auto finalize_current_pop = [&](bool useful) {
                if (!collect_cspg_stats) {
                    return;
                }
                if (useful) {
                    return;
                }
                if (current_is_routing) {
                    ++stage2_routing_useless_pops;
                } else {
                    ++stage2_nonrouting_useless_pops;
                }
            };
            if (!frontier.empty()) {
                const auto& next = frontier.front();
                if (next.partition_id < this->cspg_partition_graphs_.size()) {
                    const auto& next_graph = this->cspg_partition_graphs_[next.partition_id];
                    if (next_graph != nullptr) {
                        next_graph->Prefetch(next.id, 0);
                    }
                }
                this->basic_flatten_codes_->Prefetch(next.id);
            }
            auto result_bound = current_bound();
            expand_one_partition(current, current_is_routing, current_pop_useful, result_bound);
            finalize_current_pop(current_pop_useful);
        }
        search_result = visited_topk_results;
        if (ctx.stats != nullptr) {
            ctx.stats->hops.fetch_add(hops, std::memory_order_relaxed);
            ctx.stats->dist_cmp.fetch_add(dist_cmp, std::memory_order_relaxed);
        }
        if (collect_cspg_stats) {
            ctx.stats->cspg_stage2_hops.fetch_add(hops, std::memory_order_relaxed);
            ctx.stats->cspg_stage2_dist_cmp.fetch_add(dist_cmp, std::memory_order_relaxed);
            ctx.stats->cspg_stage2_local_dist_cmp.fetch_add(stage2_local_dist_cmp,
                                                            std::memory_order_relaxed);
            ctx.stats->cspg_stage2_cross_partition_dist_cmp.fetch_add(
                stage2_cross_partition_dist_cmp, std::memory_order_relaxed);
            ctx.stats->cspg_stage2_local_routing_dist_cmp.fetch_add(stage2_local_routing_dist_cmp,
                                                                    std::memory_order_relaxed);
            ctx.stats->cspg_stage2_local_nonrouting_dist_cmp.fetch_add(
                stage2_local_nonrouting_dist_cmp, std::memory_order_relaxed);
            ctx.stats->cspg_stage2_routing_pops.fetch_add(stage2_routing_pops,
                                                          std::memory_order_relaxed);
            ctx.stats->cspg_stage2_nonrouting_pops.fetch_add(stage2_nonrouting_pops,
                                                             std::memory_order_relaxed);
            ctx.stats->cspg_stage2_routing_useless_pops.fetch_add(stage2_routing_useless_pops,
                                                                  std::memory_order_relaxed);
            ctx.stats->cspg_stage2_nonrouting_useless_pops.fetch_add(stage2_nonrouting_useless_pops,
                                                                     std::memory_order_relaxed);
            ctx.stats->cspg_stage2_routing_fanout_attempts.fetch_add(stage2_routing_fanout_attempts,
                                                                     std::memory_order_relaxed);
            ctx.stats->cspg_stage2_routing_fanout_enqueues.fetch_add(stage2_routing_fanout_enqueues,
                                                                     std::memory_order_relaxed);
            ctx.stats->cspg_stage2_routing_fanout_skipped_unexpandable.fetch_add(
                stage2_routing_fanout_skipped_unexpandable, std::memory_order_relaxed);
            ctx.stats->cspg_stage2_routing_fanout_skipped_no_unvisited.fetch_add(
                stage2_routing_fanout_skipped_no_unvisited, std::memory_order_relaxed);
            ctx.stats->cspg_stage2_routing_fanout_skipped_duplicate_state.fetch_add(
                stage2_routing_fanout_skipped_duplicate_state, std::memory_order_relaxed);
            ctx.stats->cspg_stage2_routing_fanout_skipped_frontier_reject.fetch_add(
                stage2_routing_fanout_skipped_frontier_reject, std::memory_order_relaxed);
            ctx.stats->cspg_stage2_routing_fanout_skipped_by_bound.fetch_add(
                stage2_routing_fanout_skipped_by_bound, std::memory_order_relaxed);
        }
    } else {
        // 未启用 CSPG 时，保持原有单阶段搜索。
        search_result = this->search_one_graph(
            raw_query, this->bottom_graph_, this->basic_flatten_codes_, search_param, vt, &ctx);
    }
    this->pool_->ReturnOne(vt);

    if (use_reorder_) {
        this->reorder(raw_query, this->high_precise_codes_, search_result, k, nullptr, ctx);
    }

    while (search_result->Size() > k) {
        search_result->Pop();
    }

    // return an empty dataset directly if searcher returns nothing
    if (search_result->Empty()) {
        auto dataset_result = DatasetImpl::MakeEmptyDataset();
        dataset_result->Statistics(stats.Dump());
        return dataset_result;
    }
    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, ids] = create_fast_dataset(count, ctx.alloc);
    char* extra_infos = nullptr;
    if (extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
        extra_infos =
            static_cast<char*>(ctx.alloc->Allocate(extra_info_size_ * search_result->Size()));
        dataset_results->ExtraInfos(extra_infos);
    }
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        ids[j] = this->label_table_->GetLabelById(search_result->Top().second);
        if (extra_infos != nullptr) {
            this->extra_infos_->GetExtraInfoById(search_result->Top().second,
                                                 extra_infos + extra_info_size_ * j);
        }
        search_result->Pop();
    }
    dataset_results->Statistics(stats.Dump());
    return std::move(dataset_results);
}

void
HGraph::UpdateAttribute(int64_t id, const AttributeSet& new_attrs) {
    auto inner_id = this->label_table_->GetIdByLabel(id);
    this->attr_filter_index_->UpdateBitsetsByAttr(new_attrs, inner_id, 0);
}

void
HGraph::UpdateAttribute(int64_t id,
                        const AttributeSet& new_attrs,
                        const AttributeSet& origin_attrs) {
    auto inner_id = this->label_table_->GetIdByLabel(id);
    this->attr_filter_index_->UpdateBitsetsByAttr(new_attrs, inner_id, 0, origin_attrs);
}

const static uint64_t QUERY_SAMPLE_SIZE = 10;
const static int64_t DEFAULT_TOPK = 100;

std::string
HGraph::GetStats() const {
    AnalyzerParam analyzer_param(allocator_);
    analyzer_param.topk = DEFAULT_TOPK;
    analyzer_param.base_sample_size = std::min(QUERY_SAMPLE_SIZE, this->total_count_.load());
    analyzer_param.search_params =
        fmt::format(R"({{"hgraph": {{"ef_search": {}}}}})", ef_construct_);
    auto analyzer = CreateAnalyzer(this, analyzer_param);
    JsonType stats = analyzer->GetStats();
    return stats.Dump(4);
}

void
HGraph::init_resize_bit_and_reorder() {
    auto step_block_size = Options::Instance().block_size_limit();
    auto block_size_per_vector = this->basic_flatten_codes_->code_size_;
    block_size_per_vector =
        std::max(block_size_per_vector,
                 static_cast<uint32_t>(this->bottom_graph_->maximum_degree_ * sizeof(InnerIdType)));
    if (use_reorder_) {
        block_size_per_vector =
            std::max(block_size_per_vector, this->high_precise_codes_->code_size_);
        reorder_ = std::make_shared<FlattenReorder>(this->high_precise_codes_, allocator_);
    }
    if (this->extra_infos_ != nullptr) {
        block_size_per_vector =
            std::max<int64_t>(block_size_per_vector, static_cast<uint32_t>(this->extra_info_size_));
    }
    auto increase_count = step_block_size / block_size_per_vector;
    this->resize_increase_count_bit_ = std::max(
        DEFAULT_RESIZE_BIT, static_cast<uint64_t>(log2(static_cast<double>(increase_count))));
}

void
HGraph::check_and_init_raw_vector(const FlattenInterfaceParamPtr& raw_vector_param,
                                  const IndexCommonParam& common_param,
                                  bool is_create_new) {
    if (raw_vector_param == nullptr) {
        return;
    }

    if (is_create_new) {
        raw_vector_ = FlattenInterface::MakeInstance(raw_vector_param, common_param);
    }

    if (basic_flatten_codes_->GetQuantizerName() != QUANTIZATION_TYPE_VALUE_FP32 and
        high_precise_codes_ == nullptr) {
        create_new_raw_vector_ = true;
        has_raw_vector_ = true;
        return;
    }
    if (basic_flatten_codes_->GetQuantizerName() != QUANTIZATION_TYPE_VALUE_FP32 and
        high_precise_codes_ != nullptr and
        high_precise_codes_->GetQuantizerName() != QUANTIZATION_TYPE_VALUE_FP32) {
        create_new_raw_vector_ = true;
        has_raw_vector_ = true;
        return;
    }

    auto io_type_name = raw_vector_param->io_parameter->GetTypeName();
    if (io_type_name != IO_TYPE_VALUE_BLOCK_MEMORY_IO and io_type_name != IO_TYPE_VALUE_MEMORY_IO) {
        create_new_raw_vector_ = true;
        has_raw_vector_ = true;
        return;
    }

    if (basic_flatten_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        raw_vector_ = basic_flatten_codes_;
        has_raw_vector_ = true;
        return;
    }

    if (high_precise_codes_ != nullptr and
        high_precise_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        raw_vector_ = high_precise_codes_;
        has_raw_vector_ = true;
        return;
    }
}

bool
HGraph::UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update) {
    // check if id exists and get copied base data
    uint32_t inner_id = 0;
    {
        std::shared_lock label_lock(this->label_lookup_mutex_);
        inner_id = this->label_table_->GetIdByLabel(id);
    }

    // the validation of the new vector
    void* new_base_vec = nullptr;
    uint64_t data_size = 0;
    get_vectors(data_type_, dim_, new_base, &new_base_vec, &data_size);

    if (not force_update) {
        std::shared_lock label_lock(this->label_lookup_mutex_);

        // 1. check whether vectors are same
        Vector<int8_t> base_data(data_size, allocator_);
        GetVectorByInnerId(inner_id, (float*)base_data.data());
        float old_self_dist = this->CalcDistanceById((float*)base_data.data(), id);
        float self_dist = this->CalcDistanceById((float*)new_base_vec, id);
        if (std::abs(old_self_dist - self_dist) < 1e-3) {
            return true;
        }

        // 2. check whether the neighborhood relationship is same
        Vector<InnerIdType> neighbors(allocator_);
        this->bottom_graph_->GetNeighbors(inner_id, neighbors);
        for (auto neighbor_inner_id : neighbors) {
            // don't compare with itself
            if (neighbor_inner_id == inner_id) {
                continue;
            }

            float neighbor_dist = 0;
            try {
                neighbor_dist =
                    this->CalcDistanceById(static_cast<float*>(new_base_vec),
                                           this->label_table_->GetLabelById(neighbor_inner_id));
            } catch (const std::runtime_error& e) {
                // incase that neighbor has been deleted
                continue;
            }
            if (neighbor_dist < self_dist) {
                return false;
            }
        }
    }

    // note that only modify vector need to obtain unique lock
    // and the lock has been obtained inside datacell
    auto codes = (use_reorder_) ? high_precise_codes_ : basic_flatten_codes_;
    bool update_status = basic_flatten_codes_->UpdateVector(new_base_vec, inner_id);
    if (use_reorder_) {
        update_status = update_status && high_precise_codes_->UpdateVector(new_base_vec, inner_id);
    }
    return update_status;
}

std::string
HGraph::AnalyzeIndexBySearch(const SearchRequest& request) {
    AnalyzerParam analyzer_param(allocator_);
    analyzer_param.topk = request.topk_;
    auto analyzer = CreateAnalyzer(this, analyzer_param);
    JsonType stats = analyzer->AnalyzeIndexBySearch(request);
    return stats.Dump(4);
}

void
HGraph::GetAttributeSetByInnerId(InnerIdType inner_id, AttributeSet* attr) const {
    this->attr_filter_index_->GetAttribute(0, inner_id, attr);
}

void
HGraph::cal_memory_usage() {
    auto memory = sizeof(HGraph);
    memory += this->neighbors_mutex_->GetMemoryUsage();
    memory += this->pool_->GetMemoryUsage();
    memory += this->label_table_->GetMemoryUsage();
    memory += this->basic_flatten_codes_->GetMemoryUsage();
    memory += this->bottom_graph_->GetMemoryUsage();
    for (auto& graph : this->cspg_partition_graphs_) {
        memory += graph->GetMemoryUsage();
    }
    for (auto& route_graphs : this->cspg_partition_route_graphs_) {
        for (auto& graph : route_graphs) {
            memory += graph->GetMemoryUsage();
        }
    }
    for (auto& graph : this->route_graphs_) {
        memory += graph->GetMemoryUsage();
    }
    if (use_reorder_) {
        memory += this->high_precise_codes_->GetMemoryUsage();
    }

    if (this->extra_infos_ != nullptr and this->extra_info_size_ > 0) {
        memory += this->extra_infos_->GetMemoryUsage();
    }

    if (this->create_new_raw_vector_ and this->raw_vector_ != nullptr) {
        memory += raw_vector_->GetMemoryUsage();
    }

    std::unique_lock lock(this->memory_usage_mutex_);
    this->current_memory_usage_.store(static_cast<int64_t>(memory));
}

}  // namespace vsag
