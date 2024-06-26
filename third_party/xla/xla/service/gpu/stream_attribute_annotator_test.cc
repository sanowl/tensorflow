/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/stream_attribute_annotator.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using StreamAttributeAnnotatorTest = HloTestBase;

TEST_F(StreamAttributeAnnotatorTest, AllUsersAreAnnotated) {
  constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithAsync

  ENTRY entry {
    p1_32 = f32[1] parameter(0)
    p2_32 = f32[1] parameter(1)
    add_32 = f32[1] add(p1_32, p2_32), backend_config={"operation_queue_id":"1", "wait_on_operation_queues":[]}
    exp_32 = f32[1] exponential(add_32)

    neg32 = f32[1] negate(add_32)
    ROOT add_out_32 = f32[1] add(neg32, exp_32)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  StreamAttributeAnnotator attr_annotator;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, attr_annotator.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* add = FindInstruction(module.get(), "add_32");
  for (auto user : add->users()) {
    // Every user should have an annotation.
    EXPECT_TRUE(user->has_backend_config());
    TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                            user->backend_config<GpuBackendConfig>());
    EXPECT_EQ(gpu_config.wait_on_operation_queues()[0], 1);
  }
}

TEST_F(StreamAttributeAnnotatorTest, MultipleStreamsAreCombined) {
  constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithAsync

  ENTRY entry {
    p1_32 = f32[1] parameter(0)
    p2_32 = f32[1] parameter(1)
    add_32 = f32[1] add(p1_32, p2_32), backend_config={"operation_queue_id":"1", "wait_on_operation_queues":[]}
    exp_32 = f32[1] exponential(p2_32), backend_config={"operation_queue_id":"2", "wait_on_operation_queues":[]}

    ROOT add_out_32 = f32[1] add(add_32, exp_32)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  StreamAttributeAnnotator attr_annotator;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, attr_annotator.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  // Root should wait on 2 streams.
  EXPECT_TRUE(root->has_backend_config());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          root->backend_config<GpuBackendConfig>());
  std::vector<int64_t> expected_stream_ids = {1, 2};
  for (auto id : expected_stream_ids) {
    auto it = absl::c_find(gpu_config.wait_on_operation_queues(), id);
    EXPECT_NE(it, gpu_config.wait_on_operation_queues().end());
  }
}

TEST_F(StreamAttributeAnnotatorTest, GTEUserIsAnnotated) {
  constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithAsync

  ENTRY entry {
    p1_32 = f32[16,32] parameter(0)
    p2_32 = f32[32,16] parameter(1)

    custom-call.3 = (f32[16,16], s8[1028]{0}) custom-call(p1_32, p2_32), custom_call_target="__cublas$gemm", backend_config={"operation_queue_id":"1","wait_on_operation_queues":[],"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT","grad_x":false,"grad_y":false}}
    get-tuple-element.24 = f32[16,16] get-tuple-element(custom-call.3), index=0

    exp_32 = f32[16,16] exponential(get-tuple-element.24)

    ROOT neg32 = f32[16,16] negate(exp_32)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  StreamAttributeAnnotator attr_annotator;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, attr_annotator.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* exp = FindInstruction(module.get(), "exp_32");
  EXPECT_TRUE(exp->has_backend_config());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          exp->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.wait_on_operation_queues()[0], 1);
}

TEST_F(StreamAttributeAnnotatorTest, FusionIsAnnotated) {
  constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithFusion

  fused_computation.1 {
    fusion_p0_32 = f32[16,16] parameter(0)
    fusion_p2_32 = f32[16,16] parameter(1)
    ROOT add = f32[16,16] add(fusion_p0_32, fusion_p2_32), backend_config={"operation_queue_id":"1","wait_on_operation_queues":[]}
  }

  ENTRY entry {
    p1_32 = f32[16,16] parameter(0)
    p2_32 = f32[16,16] parameter(1)
    ROOT fusion.1 = f32[16,16] fusion(p1_32, p2_32), kind=kLoop, calls=fused_computation.1
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  StreamAttributeAnnotator attr_annotator;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, attr_annotator.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* fusion = FindInstruction(module.get(), "fusion.1");
  EXPECT_TRUE(fusion->has_backend_config());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.operation_queue_id(), 1);
}

}  // namespace
}  // namespace xla::gpu
