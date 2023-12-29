#pragma once

#include <NvInfer.h>

#include <cassert>
#include <vector>
#include <iostream>

#include "../cuda/decode.h"

using namespace nvinfer1;

#define PLUGIN_NAME "BytetrackDecode"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""

namespace rapid
{

  class DecodePlugin : public IPluginV2DynamicExt
  {
    float _score_thresh;
    int _top_n;
    // std::vector<float> _anchors;
    int _stride;

    size_t _f_width;
    size_t _f_height;
    // size_t _num_anchors;
    mutable int size = -1;

  protected:
    void deserialize(void const *data, size_t length)
    {
      const char *d = static_cast<const char *>(data);
      read(d, _score_thresh);
      read(d, _top_n);
      // size_t anchors_size;
      // read(d, anchors_size);
      // while (anchors_size--)
      // {
      //   float val;
      //   read(d, val);
      //   _anchors.push_back(val);
      // }
      read(d, _stride);
      read(d, _f_width);
      read(d, _f_height);
      // read(d, _num_anchors);
    }

    size_t getSerializationSize() const override
    {
      return sizeof(_score_thresh) + sizeof(_top_n) + sizeof(_stride) + sizeof(_f_width) + sizeof(_f_height);
    }

    void serialize(void *buffer) const override
    {
      char *d = static_cast<char *>(buffer);
      write(d, _score_thresh);
      write(d, _top_n);
      // write(d, _anchors.size());
      // for (auto &val : _anchors)
      // {
      //   write(d, val);
      // }
      write(d, _stride);
      write(d, _f_width);
      write(d, _f_height);
      // write(d, _num_anchors);
    }

  public:
    // DecodePlugin(float score_thresh, int top_n, int stride)
    //     : _score_thresh(score_thresh), _top_n(top_n),
    //       _stride(stride) {}

    DecodePlugin(float score_thresh, int top_n, int stride,
                 size_t f_width, size_t f_height)
        : _score_thresh(score_thresh), _top_n(top_n), _stride(stride), _f_width(f_width), _f_height(f_height) {}

    // Sử dụng khi load engine
    DecodePlugin(void const *data, size_t length)
    {
      this->deserialize(data, length);
    }

    const char *getPluginType() const override
    {
      return PLUGIN_NAME;
    }

    const char *getPluginVersion() const override
    {
      return PLUGIN_VERSION;
    }

    int getNbOutputs() const override
    {
      return 2;
    }

    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs *inputs,
                                  int nbInputs, IExprBuilder &exprBuilder) override
    {
      DimsExprs output(inputs[0]);
      if (outputIndex == 1)
      {
        output.d[1] = exprBuilder.constant(_top_n * 4);
      }
      else
      {
        output.d[1] = exprBuilder.constant(_top_n);
      }
      output.d[2] = exprBuilder.constant(1);
      output.d[3] = exprBuilder.constant(1);

      return output;
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut,
                                   int nbInputs, int nbOutputs) override
    {
      assert(nbInputs == 1);
      assert(nbOutputs == 2);
      assert(pos < 3);
      return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR;
    }

    int initialize() override { return 0; }

    void terminate() override {}

    size_t getWorkspaceSize(const PluginTensorDesc *inputs,
                            int nbInputs, const PluginTensorDesc *outputs, int nbOutputs) const override
    {
      if (size < 0)
      {
        size = cuda::decode(inputs->dims.d[0], nullptr, nullptr,
                            _top_n, _f_width, _f_height, _score_thresh, _stride,
                            nullptr, 0, nullptr);
      }
      return size;
    }

    int enqueue(const PluginTensorDesc *inputDesc,
                const PluginTensorDesc *outputDesc, const void *const *inputs,
                void *const *outputs, void *workspace, cudaStream_t stream)
    {

      return cuda::decode(inputDesc->dims.d[0], inputs, outputs,
                          _top_n, _f_width, _f_height, _score_thresh, _stride,
                          workspace, getWorkspaceSize(inputDesc, 1, outputDesc, 2), stream);
    }

    void destroy() override
    {
      delete this;
    };

    const char *getPluginNamespace() const override
    {
      return PLUGIN_NAMESPACE;
    }

    void setPluginNamespace(const char *N) override {}

    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const
    {
      assert(index < 2);
      return DataType::kFLOAT;
    }

    void configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs,
                         const DynamicPluginTensorDesc *out, int nbOutputs)
    {
      assert(nbInputs == 1);
      assert(nbOutputs == 2);
      // auto const &scores_dims = in[0].desc.dims;
      // auto const &boxes_dims = in[1].desc.dims;
      // assert(scores_dims.d[2] == boxes_dims.d[2]);
      // assert(scores_dims.d[3] == boxes_dims.d[3]);
      // auto const &data_dims = in[0].desc.dims;
      // _f_height = data_dims.d[2];
      // _f_width = data_dims.d[3];
    }

    IPluginV2DynamicExt *clone() const
    {
      return new DecodePlugin(_score_thresh, _top_n, _stride, _f_width, _f_height);
    }

  private:
    // đẩy data từ val vào buffer --- Trong khi save model
    template <typename T>
    void write(char *&buffer, const T &val) const
    {
      *reinterpret_cast<T *>(buffer) = val;
      buffer += sizeof(T);
    }

    // đẩy data từ buffer vào val --- Trong khi read model
    template <typename T>
    void read(const char *&buffer, T &val)
    {
      val = *reinterpret_cast<const T *>(buffer);
      buffer += sizeof(T);
    }
  };

  class DecodePluginCreator : public IPluginCreator
  {
  public:
    DecodePluginCreator() {}

    const char *getPluginName() const override
    {
      return PLUGIN_NAME;
    }

    const char *getPluginVersion() const override
    {
      return PLUGIN_VERSION;
    }

    const char *getPluginNamespace() const override
    {
      return PLUGIN_NAMESPACE;
    }

    IPluginV2DynamicExt *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override
    {
      return new DecodePlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char *N) override {}
    const PluginFieldCollection *getFieldNames() override { return nullptr; }
    IPluginV2DynamicExt *createPlugin(const char *name, const PluginFieldCollection *fc) override { return nullptr; }
  };

  REGISTER_TENSORRT_PLUGIN(DecodePluginCreator);

}

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE