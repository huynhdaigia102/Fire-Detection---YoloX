#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>
#include <initializer_list>
#include "nvdsinfer_custom_impl.h"

// #define MIN(a,b) ((a) < (b) ? (a) : (b))
/* This is a sample bounding box parsing function for the sample Resnet10
 * detector model provided with the SDK. */

struct float8 {
  float x1, y1, x2, y2, x3, y3, x4, y4;
};

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseYoloX (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferParseObjectInfo> &objectList)
{
  static int bboxLayerIndex = -1;
  static int scoresLayerIndex = -1;
  static NvDsInferDimsCHW scoresLayerDims;
  int numDetsToParse;

  /* Find the bbox layer */
  if (bboxLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "boxes") == 0) {
        bboxLayerIndex = i;
        break;
      }
    }
    if (bboxLayerIndex == -1) {
    std::cerr << "Could not find bbox layer buffer while parsing" << std::endl;
    return false;
    }
  }

  /* Find the scores layer */
  if (scoresLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "scores") == 0) {
        scoresLayerIndex = i;
        getDimsCHWFromDims(scoresLayerDims, outputLayersInfo[i].inferDims);
        break;
      }
    }
    if (scoresLayerIndex == -1) {
    std::cerr << "Could not find scores layer buffer while parsing" << std::endl;
    return false;
    }
  }

  
  /* Calculate the number of detections to parse */
  numDetsToParse = scoresLayerDims.c;

  float *bboxes = (float *) outputLayersInfo[bboxLayerIndex].buffer;
  float *scores = (float *) outputLayersInfo[scoresLayerIndex].buffer;
  
  for (int indx = 0; indx < numDetsToParse; indx++)
  {
    float x1 = bboxes[indx * 4];
    float y1 = bboxes[indx * 4 + 1];
    float x2 = bboxes[indx * 4 + 2];
    float y2 = bboxes[indx * 4 + 3];

    int this_class = 0;
    float this_score = scores[indx];
    float threshold = detectionParams.perClassThreshold[this_class];
    NvDsInferParseObjectInfo object;

    if (this_score >= 0.1) {
      object.classId = this_class;
      object.detectionConfidence = this_score;

      object.left = x1;
      object.top = y1;
      object.width = x2-x1;
      object.height = y2-y1;

      objectList.push_back(object);
    }
  }
  return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloX);
