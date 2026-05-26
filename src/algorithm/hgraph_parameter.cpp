
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

#include "hgraph_parameter.h"

#include <algorithm>
#include <cmath>

#include "datacell/extra_info_datacell_parameter.h"
#include "datacell/flatten_datacell_parameter.h"
#include "datacell/graph_datacell_parameter.h"
#include "datacell/graph_interface_parameter.h"
#include "datacell/sparse_graph_datacell_parameter.h"
#include "datacell/sparse_vector_datacell_parameter.h"
#include "impl/odescent/odescent_graph_parameter.h"
#include "inner_string_params.h"
#include "vsag/constants.h"

namespace vsag {

HGraphParameter::HGraphParameter(const JsonType& json) : HGraphParameter() {
    this->FromJson(json);
}

HGraphParameter::HGraphParameter() : name(INDEX_TYPE_HGRAPH) {
}

void
HGraphParameter::FromJson(const JsonType& json) {
    InnerIndexParameter::FromJson(json);

    if (json.Contains(HGRAPH_USE_ELP_OPTIMIZER_KEY)) {
        this->use_elp_optimizer = json[HGRAPH_USE_ELP_OPTIMIZER_KEY].GetBool();
    }

    if (json.Contains(HGRAPH_IGNORE_REORDER_KEY)) {
        this->ignore_reorder = json[HGRAPH_IGNORE_REORDER_KEY].GetBool();
    }

    if (json.Contains(HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY)) {
        this->build_by_base = json[HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY].GetBool();
    }

    CHECK_ARGUMENT(json.Contains(BASE_CODES_KEY),
                   fmt::format("hgraph parameters must contains {}", BASE_CODES_KEY));
    const auto& base_codes_json = json[BASE_CODES_KEY];
    this->base_codes_param = CreateFlattenParam(base_codes_json);

    if (use_reorder) {
        CHECK_ARGUMENT(json.Contains(PRECISE_CODES_KEY),
                       fmt::format("hgraph parameters must contains {}", PRECISE_CODES_KEY));
        const auto& precise_codes_json = json[PRECISE_CODES_KEY];
        this->precise_codes_param = CreateFlattenParam(precise_codes_json);
    }

    CHECK_ARGUMENT(json.Contains(GRAPH_KEY),
                   fmt::format("hgraph parameters must contains {}", GRAPH_KEY));
    const auto& graph_json = json[GRAPH_KEY];

    GraphStorageTypes graph_storage_type = GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT;
    if (graph_json.Contains(GRAPH_STORAGE_TYPE_KEY)) {
        const auto graph_storage_type_str = graph_json[GRAPH_STORAGE_TYPE_KEY].GetString();
        if (graph_storage_type_str == GRAPH_STORAGE_TYPE_VALUE_COMPRESSED) {
            graph_storage_type = GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_COMPRESSED;
        }

        if (graph_storage_type_str != GRAPH_STORAGE_TYPE_VALUE_COMPRESSED &&
            graph_storage_type_str != GRAPH_STORAGE_TYPE_VALUE_FLAT) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                fmt::format("invalid graph_storage_type: {}", graph_storage_type_str));
        }
    }
    this->bottom_graph_param =
        GraphInterfaceParameter::GetGraphParameterByJson(graph_storage_type, graph_json);

    hierarchical_graph_param = std::make_shared<SparseGraphDatacellParameter>();
    hierarchical_graph_param->max_degree_ = this->bottom_graph_param->max_degree_ / 2;
    if (graph_storage_type == GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT) {
        auto graph_param =
            std::dynamic_pointer_cast<GraphDataCellParameter>(this->bottom_graph_param);
        if (graph_param != nullptr) {
            hierarchical_graph_param->remove_flag_bit_ = graph_param->remove_flag_bit_;
            hierarchical_graph_param->support_delete_ = graph_param->support_remove_;
        } else {
            hierarchical_graph_param->support_delete_ = false;
        }
    } else {
        hierarchical_graph_param->support_delete_ = false;
    }

    if (json.Contains(EF_CONSTRUCTION_KEY)) {
        this->ef_construction = json[EF_CONSTRUCTION_KEY].GetInt();
    }

    if (json.Contains(ALPHA_KEY)) {
        this->alpha = json[ALPHA_KEY].GetFloat();
    }

    if (json.Contains(BUILD_THREAD_COUNT_KEY)) {
        this->build_thread_count = json[BUILD_THREAD_COUNT_KEY].GetInt();
    }

    if (graph_json.Contains(GRAPH_TYPE_KEY)) {
        graph_type = graph_json[GRAPH_TYPE_KEY].GetString();
        if (graph_type == GRAPH_TYPE_VALUE_ODESCENT) {
            odescent_param = std::make_shared<ODescentParameter>();
            odescent_param->FromJson(graph_json);
        }
    }

    if (json.Contains(SUPPORT_DUPLICATE)) {
        this->support_duplicate = json[SUPPORT_DUPLICATE].GetBool();
    }
    if (json.Contains(SUPPORT_TOMBSTONE)) {
        this->support_tombstone = json[SUPPORT_TOMBSTONE].GetBool();
    }

    // 解析 CSPG 参数并做范围校验
    if (json.Contains("cspg_m")) {
        this->cspg_m = json["cspg_m"].GetInt();
    }
    if (json.Contains("cspg_lambda")) {
        this->cspg_lambda = json["cspg_lambda"].GetFloat();
    }
    if (json.Contains(HGRAPH_CSPG_PARTITION_MAX_DEGREE)) {
        this->cspg_partition_max_degree = json[HGRAPH_CSPG_PARTITION_MAX_DEGREE].GetInt();
    }
    CHECK_ARGUMENT(this->cspg_m > 0, "cspg_m must be greater than 0");
    CHECK_ARGUMENT(this->cspg_lambda >= 0.0F && this->cspg_lambda <= 1.0F,
                   "cspg_lambda must be in range [0.0, 1.0]");
    CHECK_ARGUMENT(this->cspg_partition_max_degree >= 0,
                   "cspg_partition_max_degree must be greater than or equal to 0");
    if (this->cspg_partition_max_degree > 0) {
        CHECK_ARGUMENT(this->cspg_partition_max_degree >= 4,
                       "cspg_partition_max_degree must be 0 or at least 4");
        CHECK_ARGUMENT(static_cast<uint64_t>(this->cspg_partition_max_degree) <=
                           this->bottom_graph_param->max_degree_,
                       "cspg_partition_max_degree must not exceed max_degree");
    }
}

JsonType
HGraphParameter::ToJson() const {
    JsonType json = InnerIndexParameter::ToJson();
    json[TYPE_KEY].SetString(INDEX_TYPE_HGRAPH);

    json[HGRAPH_USE_ELP_OPTIMIZER_KEY].SetBool(this->use_elp_optimizer);
    json[HGRAPH_IGNORE_REORDER_KEY].SetBool(this->ignore_reorder);
    json[BASE_CODES_KEY].SetJson(this->base_codes_param->ToJson());
    json[GRAPH_KEY].SetJson(this->bottom_graph_param->ToJson());
    json[EF_CONSTRUCTION_KEY].SetInt(this->ef_construction);
    json[ALPHA_KEY].SetFloat(this->alpha);
    json[SUPPORT_DUPLICATE].SetBool(this->support_duplicate);
    json[TRAIN_SAMPLE_COUNT_KEY].SetInt(this->train_sample_count);
    //print new parameters
    json["cspg_m"].SetInt(this->cspg_m);
    json["cspg_lambda"].SetFloat(this->cspg_lambda);
    json[HGRAPH_CSPG_PARTITION_MAX_DEGREE].SetInt(this->cspg_partition_max_degree);
    return json;
}

bool
HGraphParameter::CheckCompatibility(const ParamPtr& other) const {
    auto hgraph_param = std::dynamic_pointer_cast<HGraphParameter>(other);
    if (hgraph_param == nullptr) {
        logger::error("HGraphParameter::CheckCompatibility: other is not HGraphParameter");
        return false;
    }
    auto have_reorder = this->use_reorder && not this->ignore_reorder;
    auto have_reorder_other = hgraph_param->use_reorder && not hgraph_param->ignore_reorder;
    if (have_reorder != have_reorder_other) {
        logger::error(
            "HGraphParameter::CheckCompatibility: use_reorder and ignore_reorder must be the same");
        return false;
    }
    if (not this->base_codes_param->CheckCompatibility(hgraph_param->base_codes_param)) {
        logger::error("HGraphParameter::CheckCompatibility: base_codes_param is not compatible");
        return false;
    }
    if (have_reorder) {
        if (not this->precise_codes_param ||
            not this->precise_codes_param->CheckCompatibility(hgraph_param->precise_codes_param)) {
            logger::error(
                "HGraphParameter::CheckCompatibility: precise_codes_param is not compatible");
            return false;
        }
    }
    if (not this->bottom_graph_param->CheckCompatibility(hgraph_param->bottom_graph_param)) {
        logger::error("HGraphParameter::CheckCompatibility: bottom_graph_param is not compatible");
        return false;
    }
    if (use_attribute_filter != hgraph_param->use_attribute_filter) {
        logger::error("HGraphParameter::CheckCompatibility: use_attribute_filter must be the same");
        return false;
    }
    if (support_duplicate != hgraph_param->support_duplicate) {
        logger::error("HGraphParameter::CheckCompatibility: support_duplicate must be the same");
        return false;
    }
    if (cspg_m != hgraph_param->cspg_m ||
        std::abs(cspg_lambda - hgraph_param->cspg_lambda) > 1e-6F ||
        cspg_partition_max_degree != hgraph_param->cspg_partition_max_degree) {
        logger::error(
            "HGraphParameter::CheckCompatibility: CSPG build parameters must be the same");
        return false;
    }
    return true;
}

uint32_t
HGraphParameter::ResolveCspgPartitionMaxDegree() const {
    if (this->bottom_graph_param == nullptr) {
        return 0;
    }
    const auto bottom_max_degree = this->bottom_graph_param->max_degree_;
    if (this->cspg_partition_max_degree > 0 || this->cspg_m <= 1) {
        return this->cspg_partition_max_degree > 0
                   ? static_cast<uint32_t>(this->cspg_partition_max_degree)
                   : static_cast<uint32_t>(bottom_max_degree);
    }

    const double partition_size_ratio =
        static_cast<double>(this->cspg_lambda) +
        (1.0 - static_cast<double>(this->cspg_lambda)) / static_cast<double>(this->cspg_m);
    const auto scaled_partition_degree = static_cast<uint64_t>(
        std::llround(static_cast<double>(bottom_max_degree) * partition_size_ratio));
    return static_cast<uint32_t>(
        std::clamp<uint64_t>(scaled_partition_degree, 1, bottom_max_degree));
}

HGraphSearchParameters
HGraphSearchParameters::FromJson(const std::string& json_string) {
    auto params = JsonType::Parse(json_string);

    HGraphSearchParameters obj;

    // set obj.ef_search
    CHECK_ARGUMENT(params.Contains(INDEX_TYPE_HGRAPH),
                   fmt::format("parameters must contains {}", INDEX_TYPE_HGRAPH));

    obj.IndexSearchParameter::FromJson(params[INDEX_TYPE_HGRAPH]);

    CHECK_ARGUMENT(
        params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_EF_RUNTIME),
        fmt::format(
            "parameters[{}] must contains {}", INDEX_TYPE_HGRAPH, HGRAPH_PARAMETER_EF_RUNTIME));
    obj.ef_search = params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_EF_RUNTIME].GetInt();
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_CSPG_EF1)) {
        obj.cspg_ef1 = params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_CSPG_EF1].GetInt();
        CHECK_ARGUMENT(obj.cspg_ef1 > 0, "cspg_ef1 must be greater than 0");
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_CSPG_EF2)) {
        obj.cspg_ef2 = params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_CSPG_EF2].GetInt();
        CHECK_ARGUMENT(obj.cspg_ef2 > 0, "cspg_ef2 must be greater than 0");
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_CSPG_PHASE1_PARTITION_COUNT)) {
        obj.cspg_phase1_partition_count =
            params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_CSPG_PHASE1_PARTITION_COUNT].GetInt();
        CHECK_ARGUMENT(obj.cspg_phase1_partition_count > 0,
                       "cspg_phase1_partition_count must be greater than 0");
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_CSPG_PHASE1_USE_ROUTE_DESCENT)) {
        obj.cspg_phase1_use_route_descent =
            params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_CSPG_PHASE1_USE_ROUTE_DESCENT].GetBool();
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_CSPG_CROSS_PARTITION_HOPS_LIMIT)) {
        obj.cspg_cross_partition_hops_limit =
            params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_CSPG_CROSS_PARTITION_HOPS_LIMIT].GetInt();
        CHECK_ARGUMENT(obj.cspg_cross_partition_hops_limit >= 0,
                       "cspg_cross_partition_hops_limit must be greater than or equal to 0");
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_CSPG_CROSS_PARTITION_SWITCH_LIMIT)) {
        obj.cspg_cross_partition_switch_limit =
            params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_CSPG_CROSS_PARTITION_SWITCH_LIMIT]
                .GetInt();
        CHECK_ARGUMENT(obj.cspg_cross_partition_switch_limit >= 0,
                       "cspg_cross_partition_switch_limit must be greater than or equal to 0");
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(
            HGRAPH_PARAMETER_CSPG_RECURSIVE_FANOUT_BOUND_SLACK_PERCENT)) {
        obj.cspg_recursive_fanout_bound_slack_percent =
            params[INDEX_TYPE_HGRAPH]
                  [HGRAPH_PARAMETER_CSPG_RECURSIVE_FANOUT_BOUND_SLACK_PERCENT]
                      .GetInt();
        CHECK_ARGUMENT(obj.cspg_recursive_fanout_bound_slack_percent >= 0 &&
                           obj.cspg_recursive_fanout_bound_slack_percent <= 100,
                       "cspg_recursive_fanout_bound_slack_percent must be between 0 and 100");
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_CSPG_LOCAL_ROUTING_BUDGET)) {
        obj.cspg_local_routing_budget =
            params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_CSPG_LOCAL_ROUTING_BUDGET].GetInt();
        CHECK_ARGUMENT(obj.cspg_local_routing_budget >= 0,
                       "cspg_local_routing_budget must be greater than or equal to 0");
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_CSPG_ENABLE_STATS)) {
        obj.cspg_enable_stats =
            params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_CSPG_ENABLE_STATS].GetBool();
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_HOPS_LIMIT)) {
        obj.hops_limit = params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_HOPS_LIMIT].GetInt();
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_USE_EXTRA_INFO_FILTER)) {
        obj.use_extra_info_filter =
            params[INDEX_TYPE_HGRAPH][HGRAPH_USE_EXTRA_INFO_FILTER].GetBool();
    }

    return obj;
}
}  // namespace vsag
