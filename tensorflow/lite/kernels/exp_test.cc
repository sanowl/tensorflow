/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <math.h>

#include <cmath>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "Eigen/Core"  // from @eigen_archive
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/types/half.h"

namespace tflite {

namespace ops {
namespace builtin {

TfLiteRegistration* Register_EXP_REF();
TfLiteRegistration* Register_EXP_GENERIC_OPT();

}  // namespace builtin
}  // namespace ops

namespace {

using ::testing::ElementsAreArray;

class BaseExpOpModel : public SingleOpModel {
 public:
  // With no registration, the default builtin Exp kernel is used.
  BaseExpOpModel(const TensorData& input, const TensorData& output,
                 TfLiteRegistration* registration = nullptr) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_EXP, BuiltinOptions_ExpOptions,
                 CreateExpOptions(builder_).Union());
    if (registration != nullptr) {
      resolver_ =
          std::make_unique<SingleOpResolver>(BuiltinOperator_EXP, registration);
    }
    BuildInterpreter({GetShape(input_)});
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int output_;
};

class FloatExpOpModel : public BaseExpOpModel {
 public:
  using BaseExpOpModel::BaseExpOpModel;

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }
  void PopulateInput(const std::vector<float>& data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class QuantizedExpOpModel : public BaseExpOpModel {
 public:
  using BaseExpOpModel::BaseExpOpModel;

  template <class T>
  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }
};

// A LUT of 256 values is used in the int8 case and 512 values (+1 for
// interpolation) for the int16 case
template <typename T>
inline float GetTolerance(float min, float max) {
  float kQuantizedTolerance = (max - min) / (std::numeric_limits<T>::max() -
                                             std::numeric_limits<T>::min());
  if (std::is_same<T, int8_t>::value) {
    kQuantizedTolerance += (max - min) / 256.0f;
  } else if (std::is_same<T, int16_t>::value) {
    kQuantizedTolerance += (max - min) / 512.0f;
  }

  return kQuantizedTolerance;
}

TEST(ExpOpTest, ExpFloat) {
  std::initializer_list<float> data = {0.0f,    1.0f,  -1.0f, 100.0f,
                                       -100.0f, 0.01f, -0.01f};
  FloatExpOpModel m({TensorType_FLOAT32, {1, 1, 7}}, {TensorType_FLOAT32, {}});
  m.SetInput(data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 7}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {std::exp(0.0f), std::exp(1.0f), std::exp(-1.0f), std::exp(100.0f),
           std::exp(-100.0f), std::exp(0.01f), std::exp(-0.01f)})));
}

const auto kExpKernelMap = new std::map<std::string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_EXP_REF()},
    {"GenericOptimized", ops::builtin::Register_EXP_GENERIC_OPT()},
});

class ExpKernelTest : public SingleOpTest {
 protected:
  const std::map<std::string, TfLiteRegistration*>& GetKernelMap() override {
    return *kExpKernelMap;
  }
};

// Long enough to exercise both the vectorized body and the scalar tail of the
// optimized kernel, and spans overflow, underflow and denormal results.
TEST_P(ExpKernelTest, MatchesStdExpOverWideRange) {
  constexpr int kSize = 1027;
  FloatExpOpModel m({TensorType_FLOAT32, {kSize}}, {TensorType_FLOAT32, {}},
                    GetRegistration());
  std::vector<float> input(kSize);
  for (int i = 0; i < kSize; ++i) {
    input[i] = -110.0f + 200.0f * i / (kSize - 1);
  }
  m.PopulateInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const std::vector<float> output = m.GetOutput();
  ASSERT_EQ(output.size(), kSize);
  for (int i = 0; i < kSize; ++i) {
    const float expected = std::exp(input[i]);
    if (std::isinf(expected) || expected == 0.0f) {
      EXPECT_EQ(output[i], expected) << "x = " << input[i];
    } else {
      // Two ulps of relative error, plus slack for denormal results.
      EXPECT_NEAR(output[i], expected,
                  2.4e-7f * expected + std::numeric_limits<float>::denorm_min())
          << "x = " << input[i];
    }
  }
}

TEST_P(ExpKernelTest, SpecialValues) {
  FloatExpOpModel m({TensorType_FLOAT32, {4}}, {TensorType_FLOAT32, {}},
                    GetRegistration());
  m.SetInput({std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity(), 0.0f,
              std::numeric_limits<float>::quiet_NaN()});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const std::vector<float> output = m.GetOutput();
  EXPECT_TRUE(std::isinf(output[0]) && output[0] > 0);
  EXPECT_EQ(output[1], 0.0f);
  EXPECT_EQ(output[2], 1.0f);
  EXPECT_TRUE(std::isnan(output[3]));
}

INSTANTIATE_TEST_SUITE_P(
    ExpKernelTest, ExpKernelTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kExpKernelMap)));

template <typename T>
class ReducedPrecisionExpOpModel : public BaseExpOpModel {
 public:
  using BaseExpOpModel::BaseExpOpModel;

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor<T>(input_, data);
  }

  std::vector<float> GetOutputAsFloat() {
    std::vector<float> result;
    for (const T& value : ExtractVector<T>(output_)) {
      result.push_back(static_cast<float>(value));
    }
    return result;
  }
};

// All inputs below are exactly representable in fp16 and bf16, so the only
// error is rounding the result. The tolerances cover half an ulp at e^2.
TEST(ExpOpTest, ExpFloat16) {
  ReducedPrecisionExpOpModel<half> m({TensorType_FLOAT16, {1, 1, 7}},
                                     {TensorType_FLOAT16, {}});
  m.SetInput({0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 7}));
  EXPECT_THAT(m.GetOutputAsFloat(),
              ElementsAreArray(ArrayFloatNear(
                  {std::exp(0.0f), std::exp(1.0f), std::exp(-1.0f),
                   std::exp(2.0f), std::exp(-2.0f), std::exp(0.5f),
                   std::exp(-0.5f)},
                  /*max_abs_error=*/4e-3f)));
}

TEST(ExpOpTest, ExpFloat16OverflowsToInfinity) {
  // The largest finite fp16 value is 65504, and e^12 is about 162755.
  ReducedPrecisionExpOpModel<half> m({TensorType_FLOAT16, {2}},
                                     {TensorType_FLOAT16, {}});
  m.SetInput({12.0f, -20.0f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const std::vector<float> output = m.GetOutputAsFloat();
  EXPECT_TRUE(std::isinf(output[0]));
  EXPECT_EQ(output[1], 0.0f);
}

TEST(ExpOpTest, ExpBFloat16) {
  ReducedPrecisionExpOpModel<Eigen::bfloat16> m(
      {TensorType_BFLOAT16, {1, 1, 7}}, {TensorType_BFLOAT16, {}});
  m.SetInput({0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 7}));
  EXPECT_THAT(m.GetOutputAsFloat(),
              ElementsAreArray(ArrayFloatNear(
                  {std::exp(0.0f), std::exp(1.0f), std::exp(-1.0f),
                   std::exp(2.0f), std::exp(-2.0f), std::exp(0.5f),
                   std::exp(-0.5f)},
                  /*max_abs_error=*/2e-2f)));
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedExpSymmetricTest() {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);
  const float kQuantizedTolerance = GetTolerance<integer_dtype>(-3.1, 3.1);
  QuantizedExpOpModel m({tensor_type, {1, 2, 2, 2}, 1.3f * kMin, 1.3f * kMax},
                        {tensor_type, {}, 3.01f * kMin, 3.01f * kMax});
  m.SetInput<integer_dtype>({-1.3, -1.0, -0.3, 0, 0.1, 0.5, 1.0, 1.1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 2, 2}));
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear(
                  {0.2725, 0.3679, 0.7408, 1.0, 1.1052, 1.6487, 2.7183, 3.0042},
                  kQuantizedTolerance)));
}

TEST(ExpOpTest, ExpSymmetricInt8) {
  QuantizedExpSymmetricTest<TensorType_INT8, int8_t>();
}

TEST(ExpOpTest, ExpSymmetricInt16) {
  QuantizedExpSymmetricTest<TensorType_INT16, int16_t>();
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedExpAsymmetricTest() {
  const float kQuantizedTolerance = GetTolerance<integer_dtype>(-1.3, 3.01);
  QuantizedExpOpModel m({tensor_type, {1, 2, 2, 2}, -1.3, 1.1},
                        {tensor_type, {}, 0.0, 3.01});
  m.SetInput<integer_dtype>({-1.3, -1.0, -0.3, 0, 0.1, 0.5, 1.0, 1.1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 2, 2}));
  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear(
                  {0.2725, 0.3679, 0.7408, 1.0, 1.1052, 1.6487, 2.7183, 3.0042},
                  kQuantizedTolerance)));
}

TEST(ExpOpTest, ExpAsymmetricInt8) {
  QuantizedExpAsymmetricTest<TensorType_INT8, int8_t>();
}

}  // namespace
}  // namespace tflite
