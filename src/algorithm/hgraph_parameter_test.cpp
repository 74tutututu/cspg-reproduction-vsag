
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

#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "parameter_test.h"

#define TEST_COMPATIBILITY_CASE(section_name, param_member, val1, val2, expect_compatible) \
    SECTION(section_name) {                                                                \
        HGraphDefaultParam param1;                                                         \
        HGraphDefaultParam param2;                                                         \
        param1.param_member = val1;                                                        \
        param2.param_member = val2;                                                        \
        auto param_str1 = generate_hgraph_param(param1);                                   \
        auto param_str2 = generate_hgraph_param(param2);                                   \
        auto hgraph_param1 = std::make_shared<vsag::HGraphParameter>();                    \
        auto hgraph_param2 = std::make_shared<vsag::HGraphParameter>();                    \
        hgraph_param1->FromString(param_str1);                                             \
        hgraph_param2->FromString(param_str2);                                             \
        if (expect_compatible) {                                                           \
            REQUIRE(hgraph_param1->CheckCompatibility(hgraph_param2));                     \
        } else {                                                                           \
            REQUIRE_FALSE(hgraph_param1->CheckCompatibility(hgraph_param2));               \
        }                                                                                  \
    }

struct HGraphDefaultParam {
    std::string base_codes_io_type = "block_memory_io";
    std::string base_codes_quantization_type = "pq";
    int base_codes_pq_dim = 8;
    std::string precise_codes_io_type = "block_memory_io";
    std::string graph_io_type = "block_memory_io";
    std::string graph_storage_type = "flat";
    std::string precise_codes_quantization_type = "fp32";
    int max_degree = 26;
    bool support_remove = true;
    int remove_flag_bit = 8;
    bool use_attribute_filter = false;
    bool support_duplicate = false;
    bool use_reorder = true;
    int cspg_m = 1;
    float cspg_lambda = 0.5F;
    int cspg_partition_max_degree = 0;
};

std::string
generate_hgraph_param(const HGraphDefaultParam& param) {
    static constexpr auto param_str = R"({{
        "base_codes": {{
            "codes_type": "flatten_codes",
            "io_params": {{
                "file_path": "./default_file_path",
                "type": "{}"
            }},
            "quantization_params": {{
                "pca_dim": 0,
                "pq_dim": {},
                "sq4_uniform_trunc_rate": 0.05,
                "type": "{}"
            }}
        }},
        "build_by_base": false,
        "extra_info": {{
            "io_params": {{
                "file_path": "./default_file_path",
                "type": "block_memory_io"
            }}
        }},
        "graph": {{
            "graph_storage_type": "{}",
            "init_capacity": 100,
            "io_params": {{
                "file_path": "./default_file_path",
                "type": "block_memory_io"
            }},
            "max_degree": {},
            "support_remove": {},
            "remove_flag_bit": {}
        }},
        "ignore_reorder": false,
        "precise_codes": {{
            "codes_type": "flatten_codes",
            "io_params": {{
                "file_path": "./default_file_path",
                "type": "{}"
            }},
            "quantization_params": {{
                "pca_dim": 0,
                "pq_dim": 1,
                "sq4_uniform_trunc_rate": 0.05,
                "type": "{}"
            }}
        }},
        "type": "hgraph",
        "use_attribute_filter": {},
        "use_reorder": {},
        "support_duplicate": {},
        "cspg_m": {},
        "cspg_lambda": {},
        "cspg_partition_max_degree": {}
    }})";

    return fmt::format(param_str,
                       param.base_codes_io_type,
                       param.base_codes_pq_dim,
                       param.base_codes_quantization_type,
                       param.graph_storage_type,
                       param.max_degree,
                       param.support_remove,
                       param.remove_flag_bit,
                       param.precise_codes_io_type,
                       param.precise_codes_quantization_type,
                       param.use_attribute_filter,
                       param.use_reorder,
                       param.support_duplicate,
                       param.cspg_m,
                       param.cspg_lambda,
                       param.cspg_partition_max_degree);
}

TEST_CASE("HGraph Parameters CheckCompatibility", "[ut][HGraphParameter][CheckCompatibility]"){
    SECTION("cspg disabled by default"){vsag::HGraphParameter param;
REQUIRE(param.cspg_m == 1);
REQUIRE(std::abs(param.cspg_lambda - 0.5F) < 1e-6F);
REQUIRE(param.cspg_partition_max_degree == 0);
}

SECTION("wrong parameter type") {
    HGraphDefaultParam default_param;
    auto param_str = generate_hgraph_param(default_param);
    auto param = std::make_shared<vsag::HGraphParameter>();
    param->FromString(param_str);
    REQUIRE(param->CheckCompatibility(param));
    REQUIRE_FALSE(param->CheckCompatibility(std::make_shared<vsag::EmptyParameter>()));
}

TEST_COMPATIBILITY_CASE(
    "different base codes io type", base_codes_io_type, "memory_io", "block_memory_io", true)
TEST_COMPATIBILITY_CASE("different pq dim", base_codes_pq_dim, 8, 16, false)
TEST_COMPATIBILITY_CASE(
    "different base codes quantization type", base_codes_quantization_type, "sq4", "sq8", false)
TEST_COMPATIBILITY_CASE("different graph type", graph_storage_type, "flat", "compressed", false)
TEST_COMPATIBILITY_CASE("different max degree", max_degree, 26, 30, false)
TEST_COMPATIBILITY_CASE("different support remove", support_remove, true, false, false)
TEST_COMPATIBILITY_CASE("different remove flag bit", remove_flag_bit, 8, 16, false)
TEST_COMPATIBILITY_CASE("different use reorder", use_reorder, true, false, false)
TEST_COMPATIBILITY_CASE(
    "different precise codes io type", precise_codes_io_type, "memory_io", "block_memory_io", true)
TEST_COMPATIBILITY_CASE("different precise codes quantization type",
                        precise_codes_quantization_type,
                        "fp32",
                        "sq8",
                        false)
TEST_COMPATIBILITY_CASE("different use attribute filter", use_attribute_filter, true, false, false)
TEST_COMPATIBILITY_CASE("different support duplicate", support_duplicate, true, false, false)
TEST_COMPATIBILITY_CASE("different cspg_m", cspg_m, 2, 4, false)
TEST_COMPATIBILITY_CASE("different cspg_lambda", cspg_lambda, 0.5F, 0.3F, false)
TEST_COMPATIBILITY_CASE(
    "different cspg_partition_max_degree", cspg_partition_max_degree, 16, 24, false)
}

TEST_CASE("HGraph Parameters ResolveCspgPartitionMaxDegree", "[ut][HGraphParameter][ResolveCspgPartitionMaxDegree]") {
    SECTION("keep explicit partition degree") {
        HGraphDefaultParam default_param;
        default_param.max_degree = 32;
        default_param.cspg_m = 2;
        default_param.cspg_lambda = 0.5F;
        default_param.cspg_partition_max_degree = 28;
        auto param = std::make_shared<vsag::HGraphParameter>();
        param->FromString(generate_hgraph_param(default_param));
        REQUIRE(param->ResolveCspgPartitionMaxDegree() == 28);
    }

    SECTION("scale default partition degree by partition size ratio") {
        HGraphDefaultParam default_param;
        default_param.max_degree = 32;
        default_param.cspg_m = 2;
        default_param.cspg_lambda = 0.5F;
        default_param.cspg_partition_max_degree = 0;
        auto param = std::make_shared<vsag::HGraphParameter>();
        param->FromString(generate_hgraph_param(default_param));
        REQUIRE(param->ResolveCspgPartitionMaxDegree() == 24);
    }

    SECTION("disable scaling when cspg is off") {
        HGraphDefaultParam default_param;
        default_param.max_degree = 32;
        default_param.cspg_m = 1;
        default_param.cspg_partition_max_degree = 0;
        auto param = std::make_shared<vsag::HGraphParameter>();
        param->FromString(generate_hgraph_param(default_param));
        REQUIRE(param->ResolveCspgPartitionMaxDegree() == 32);
    }
}

TEST_CASE("HGraph Search Parameters Parse CSPG", "[ut][HGraphParameter][Search]") {
    SECTION("parse cspg ef1") {
        auto params =
            vsag::HGraphSearchParameters::FromJson(R"({"hgraph":{"ef_search":60,"cspg_ef1":1}})");
        REQUIRE(params.ef_search == 60);
        REQUIRE(params.cspg_ef1 == 1);
        REQUIRE(params.cspg_ef2 == 0);
    }

    SECTION("parse cspg ef2") {
        auto params = vsag::HGraphSearchParameters::FromJson(
            R"({"hgraph":{"ef_search":60,"cspg_ef1":1,"cspg_ef2":30}})");
        REQUIRE(params.ef_search == 60);
        REQUIRE(params.cspg_ef1 == 1);
        REQUIRE(params.cspg_ef2 == 30);
        REQUIRE(params.cspg_phase1_partition_count == 1);
        REQUIRE_FALSE(params.cspg_enable_stats);
    }

    SECTION("parse cspg phase1 partition count") {
        auto params = vsag::HGraphSearchParameters::FromJson(
            R"({"hgraph":{"ef_search":60,"cspg_ef1":1,"cspg_ef2":30,"cspg_phase1_partition_count":2}})");
        REQUIRE(params.cspg_phase1_partition_count == 2);
        REQUIRE(params.cspg_phase1_use_route_descent);
    }

    SECTION("parse cspg phase1 route descent") {
        auto params = vsag::HGraphSearchParameters::FromJson(
            R"({"hgraph":{"ef_search":60,"cspg_ef1":1,"cspg_ef2":30,"cspg_phase1_use_route_descent":true}})");
        REQUIRE(params.cspg_phase1_use_route_descent);
    }

    SECTION("parse cspg disable phase1 route descent") {
        auto params = vsag::HGraphSearchParameters::FromJson(
            R"({"hgraph":{"ef_search":60,"cspg_ef1":1,"cspg_ef2":30,"cspg_phase1_use_route_descent":false}})");
        REQUIRE_FALSE(params.cspg_phase1_use_route_descent);
    }

    SECTION("parse cspg cross partition hops limit") {
        auto params = vsag::HGraphSearchParameters::FromJson(
            R"({"hgraph":{"ef_search":60,"cspg_ef1":1,"cspg_ef2":30,"cspg_cross_partition_hops_limit":2}})");
        REQUIRE(params.cspg_cross_partition_hops_limit == 2);
    }

    SECTION("parse cspg cross partition switch limit") {
        auto params = vsag::HGraphSearchParameters::FromJson(
            R"({"hgraph":{"ef_search":60,"cspg_ef1":1,"cspg_ef2":30,"cspg_cross_partition_switch_limit":1}})");
        REQUIRE(params.cspg_cross_partition_switch_limit == 1);
    }

    SECTION("parse cspg recursive fanout bound slack percent") {
        auto params = vsag::HGraphSearchParameters::FromJson(
            R"({"hgraph":{"ef_search":60,"cspg_ef1":1,"cspg_ef2":30,"cspg_recursive_fanout_bound_slack_percent":75}})");
        REQUIRE(params.cspg_recursive_fanout_bound_slack_percent == 75);
    }

    SECTION("parse cspg stats switch") {
        auto params = vsag::HGraphSearchParameters::FromJson(
            R"({"hgraph":{"ef_search":60,"cspg_ef1":1,"cspg_ef2":30,"cspg_enable_stats":true}})");
        REQUIRE(params.cspg_enable_stats);
    }

    SECTION("parse cspg local routing budget") {
        auto params = vsag::HGraphSearchParameters::FromJson(
            R"({"hgraph":{"ef_search":60,"cspg_ef1":1,"cspg_ef2":30,"cspg_local_routing_budget":4}})");
        REQUIRE(params.cspg_local_routing_budget == 4);
    }

    SECTION("reject invalid cspg ef1") {
        REQUIRE_THROWS(
            vsag::HGraphSearchParameters::FromJson(R"({"hgraph":{"ef_search":60,"cspg_ef1":0}})"));
    }

    SECTION("reject invalid cspg ef2") {
        REQUIRE_THROWS(vsag::HGraphSearchParameters::FromJson(
            R"({"hgraph":{"ef_search":60,"cspg_ef1":1,"cspg_ef2":0}})"));
    }

    SECTION("reject invalid cspg phase1 partition count") {
        REQUIRE_THROWS(vsag::HGraphSearchParameters::FromJson(
            R"({"hgraph":{"ef_search":60,"cspg_ef1":1,"cspg_phase1_partition_count":0}})"));
    }

    SECTION("reject invalid cspg cross partition hops limit") {
        REQUIRE_THROWS(vsag::HGraphSearchParameters::FromJson(
            R"({"hgraph":{"ef_search":60,"cspg_ef1":1,"cspg_cross_partition_hops_limit":-1}})"));
    }

    SECTION("reject invalid cspg cross partition switch limit") {
        REQUIRE_THROWS(vsag::HGraphSearchParameters::FromJson(
            R"({"hgraph":{"ef_search":60,"cspg_ef1":1,"cspg_cross_partition_switch_limit":-1}})"));
    }

    SECTION("reject invalid cspg recursive fanout bound slack percent") {
        REQUIRE_THROWS(vsag::HGraphSearchParameters::FromJson(
            R"({"hgraph":{"ef_search":60,"cspg_ef1":1,"cspg_recursive_fanout_bound_slack_percent":101}})"));
    }

    SECTION("reject invalid cspg local routing budget") {
        REQUIRE_THROWS(vsag::HGraphSearchParameters::FromJson(
            R"({"hgraph":{"ef_search":60,"cspg_ef1":1,"cspg_local_routing_budget":-1}})"));
    }
}
