
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

#include <limits>

#include "data_type.h"
#include "index_search_parameter.h"
#include "inner_index_parameter.h"
#include "utils/pointer_define.h"
#include "vsag/constants.h"

namespace vsag {
DEFINE_POINTER2(ExtraInfoDataCellParam, ExtraInfoDataCellParameter);
DEFINE_POINTER2(FlattenInterfaceParam, FlattenInterfaceParameter);
DEFINE_POINTER2(GraphInterfaceParam, GraphInterfaceParameter);
DEFINE_POINTER2(SparseGraphDatacellParam, SparseGraphDatacellParameter);
DEFINE_POINTER(ODescentParameter);

DEFINE_POINTER(HGraphParameter);
class HGraphParameter : public InnerIndexParameter {
public:
    explicit HGraphParameter(const JsonType& json);

    HGraphParameter();

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const ParamPtr& other) const override;

    [[nodiscard]] uint32_t
    ResolveCspgPartitionMaxDegree() const;

public:
    FlattenInterfaceParamPtr base_codes_param{nullptr};
    GraphInterfaceParamPtr bottom_graph_param{nullptr};
    SparseGraphDatacellParamPtr hierarchical_graph_param{nullptr};

    ODescentParameterPtr odescent_param{nullptr};

    std::string graph_type{GRAPH_TYPE_VALUE_NSW};

    bool use_elp_optimizer{false};
    bool ignore_reorder{false};
    bool build_by_base{false};

    uint64_t ef_construction{400};
    float alpha{1.0F};

    bool support_duplicate{false};
    bool support_tombstone{false};

    DataTypes data_type{DataTypes::DATA_TYPE_FLOAT};

    // CSPG 默认关闭，显式设置 cspg_m > 1 时开启。
    int cspg_m{1};
    float cspg_lambda{0.5F};
    int cspg_partition_max_degree{0};
    // CSPG partition graph 构图类型，默认 "odescent"；显式设置 "nsw" 可走 NSW 插入路径。
    std::string cspg_partition_graph_type{"odescent"};
    // ===========================

    std::string name;
};

class HGraphSearchParameters : public IndexSearchParameter {
public:
    static HGraphSearchParameters
    FromJson(const std::string& json_string);

public:
    int64_t ef_search{30};
    int64_t cspg_ef1{1};  // CSPG 第一阶段候选池大小
    int64_t cspg_ef2{0};  // CSPG 第二阶段候选池大小，0 表示复用 ef_search
    int64_t cspg_phase1_partition_count{1};
    // 默认贴近论文 Algorithm 1，不先走 route graph。
    // 需要 HGraph 风格增强路径时可显式设为 true.
    bool cspg_phase1_use_route_descent{false};
    // When route descent is on, skip the redundant flat base-layer beam in
    // phase-1: stop the route descent above level 0 and hand the route entry
    // straight to phase-2, which performs the single base-layer search. This
    // removes the duplicated base-layer descent that phase-2 would otherwise
    // repeat after the visited reset.
    bool cspg_phase1_skip_base_descent{false};
    int64_t cspg_cross_partition_hops_limit{0};
    int64_t cspg_cross_partition_switch_limit{0};
    int64_t cspg_recursive_fanout_bound_slack_percent{0};
    int64_t cspg_local_routing_budget{0};
    bool cspg_enable_stats{false};
    // 每次遇到 routing vector 时最多向几个其他 partition 添加状态；0 = 不限制（论文行为）。
    int64_t cspg_max_routing_fanout{0};
    uint32_t hops_limit{std::numeric_limits<uint32_t>::max()};
    bool use_reorder{false};
    bool use_extra_info_filter{false};

private:
    HGraphSearchParameters() = default;
};

}  // namespace vsag
