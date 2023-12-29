/* GStreamer
 * Copyright (C) 2022 FIXME <fixme@example.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Suite 500,
 * Boston, MA 02110-1335, USA.
 */
/**
 * SECTION:element-gstyoloxparser
 *
 * The yoloxparser element does FIXME stuff.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 -v fakesrc ! yoloxparser ! FIXME ! fakesink
 * ]|
 * FIXME Describe what the pipeline does.
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <algorithm>
#include <iostream>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <math.h>

#include "gstyoloxparser.h"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
// #include "cuda_runtime_api.h"
#include "nvdsinfer_custom_impl.h"
#include "yolox_meta.h"
#include "nvdsmeta.h"
#include "nvdsmeta_schema.h"

GST_DEBUG_CATEGORY_STATIC (gst_yoloxparser_debug_category);
#define GST_CAT_DEFAULT gst_yoloxparser_debug_category

gpointer box_meta_copy_func (gpointer data, gpointer user_data) {
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  FisheyeData *srcMeta = (FisheyeData *) user_meta->user_meta_data;
  FisheyeData *dstMeta = NULL;

  dstMeta = (FisheyeData *)g_memdup (srcMeta, sizeof(FisheyeData));

  dstMeta->foot_bbox = srcMeta->foot_bbox;
  dstMeta->full_bbox = srcMeta->full_bbox;

  return dstMeta;
}

void box_meta_free_func (gpointer data, gpointer user_data) {
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  FisheyeData *srcMeta = (FisheyeData *) user_meta->user_meta_data;

  g_free (user_meta->user_meta_data);
  user_meta->user_meta_data = NULL;
}

YoloxType NvDsInferParseYoloX (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<YoloXObjectDetectionInfo> &objectList)
{
  static int bboxLayerIndex = -1;
  static int scoresLayerIndex = -1;
  static NvDsInferDimsCHW scoresLayerDims;
  static NvDsInferDimsCHW boxesLayerDims;
  int numDetsToParse;

  /* Find the bbox layer */
  if (bboxLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "boxes") == 0) {
        bboxLayerIndex = i;
        getDimsCHWFromDims(boxesLayerDims, outputLayersInfo[i].inferDims);
        break;
      }
    }
    if (bboxLayerIndex == -1) {
    std::cerr << "Could not find bbox layer buffer while parsing" << std::endl;
    return YoloX_None;
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
    return YoloX_None;
    }
  }

  
  /* Calculate the number of detections to parse */
  numDetsToParse = scoresLayerDims.c;
  YoloxType yolox_type = ((boxesLayerDims.c / numDetsToParse) == 5) ? YoloX_Fisheye : YoloX_Normal;

  float *bboxes = (float *) outputLayersInfo[bboxLayerIndex].buffer;
  float *scores = (float *) outputLayersInfo[scoresLayerIndex].buffer;
  
  for (int indx = 0; indx < numDetsToParse; indx++)
  {
    YoloXObjectDetectionInfo object;

    if (yolox_type == YoloX_Fisheye) {
      object.left = bboxes[indx * 5] - bboxes[indx * 5 + 2]/2;
      object.top = bboxes[indx * 5 + 1] - bboxes[indx * 5 + 3]/2;
      object.width = bboxes[indx * 5 + 2];
      object.height = bboxes[indx * 5 + 3];

      // foots point
      float angle = bboxes[indx * 5 + 4];
      float fx = bboxes[indx * 5] - object.height / 2. * std::sin(angle);
      float fy = bboxes[indx * 5 + 1] + object.height / 2. * std::cos(angle);
      object.foot_pt = Point(fx, fy);
      object.left = fx - object.width / 2.;
      object.top = fy - object.height;

      float cosTheta2 = std::cos(angle) * 0.5f;
      float sinTheta2 = std::sin(angle) * -0.5f;
      object.pts[0].x = bboxes[indx * 5] + sinTheta2 * object.height + cosTheta2 * object.width;
      object.pts[0].y = bboxes[indx * 5 + 1] + cosTheta2 * object.height - sinTheta2 * object.width;
      object.pts[1].x = bboxes[indx * 5] - sinTheta2 * object.height + cosTheta2 * object.width;
      object.pts[1].y = bboxes[indx * 5 + 1] - cosTheta2 * object.height - sinTheta2 * object.width;
      object.pts[2].x = 2 * bboxes[indx * 5] - object.pts[0].x;
      object.pts[2].y = 2 * bboxes[indx * 5 + 1] - object.pts[0].y;
      object.pts[3].x = 2 * bboxes[indx * 5] - object.pts[1].x;
      object.pts[3].y = 2 * bboxes[indx * 5 + 1] - object.pts[1].y;

      // object.left = std::min({object.pts[0].x, object.pts[1].x, object.pts[2].x, object.pts[3].x});
      // object.top = std::min({object.pts[0].y, object.pts[1].y, object.pts[2].y, object.pts[3].y});
      // object.width = std::max({object.pts[0].x, object.pts[1].x, object.pts[2].x, object.pts[3].x}) - object.left;
      // object.height = std::max({object.pts[0].y, object.pts[1].y, object.pts[2].y, object.pts[3].y}) - object.top;

    } else {
      object.left = bboxes[indx * 4];
      object.top = bboxes[indx * 4 + 1];
      object.width = bboxes[indx * 4 + 2] - bboxes[indx * 4];
      object.height = bboxes[indx * 4 + 3] - bboxes[indx * 4 + 1];
    }

    int this_class = 0;
    float this_score = scores[indx];
    float threshold = detectionParams.perClassThreshold[this_class];

    if (this_score >= threshold) {
      object.classId = this_class;
      object.detectionConfidence = this_score;

      objectList.push_back(object);
    }
  }
  return yolox_type;
}

/* prototypes */


static void gst_yoloxparser_set_property (GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_yoloxparser_get_property (GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);
static void gst_yoloxparser_dispose (GObject * object);
static void gst_yoloxparser_finalize (GObject * object);

static GstCaps *gst_yoloxparser_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter);
static GstCaps *gst_yoloxparser_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_yoloxparser_accept_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps);
static gboolean gst_yoloxparser_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_yoloxparser_query (GstBaseTransform * trans,
    GstPadDirection direction, GstQuery * query);
static gboolean gst_yoloxparser_decide_allocation (GstBaseTransform * trans,
    GstQuery * query);
static gboolean gst_yoloxparser_filter_meta (GstBaseTransform * trans,
    GstQuery * query, GType api, const GstStructure * params);
static gboolean gst_yoloxparser_propose_allocation (GstBaseTransform * trans,
    GstQuery * decide_query, GstQuery * query);
static gboolean gst_yoloxparser_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size, GstCaps * othercaps,
    gsize * othersize);
static gboolean gst_yoloxparser_get_unit_size (GstBaseTransform * trans,
    GstCaps * caps, gsize * size);
static gboolean gst_yoloxparser_start (GstBaseTransform * trans);
static gboolean gst_yoloxparser_stop (GstBaseTransform * trans);
static gboolean gst_yoloxparser_sink_event (GstBaseTransform * trans,
    GstEvent * event);
static gboolean gst_yoloxparser_src_event (GstBaseTransform * trans,
    GstEvent * event);
static GstFlowReturn gst_yoloxparser_prepare_output_buffer (GstBaseTransform *
    trans, GstBuffer * input, GstBuffer ** outbuf);
static gboolean gst_yoloxparser_copy_metadata (GstBaseTransform * trans,
    GstBuffer * input, GstBuffer * outbuf);
static gboolean gst_yoloxparser_transform_meta (GstBaseTransform * trans,
    GstBuffer * outbuf, GstMeta * meta, GstBuffer * inbuf);
static void gst_yoloxparser_before_transform (GstBaseTransform * trans,
    GstBuffer * buffer);
static GstFlowReturn gst_yoloxparser_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf);
static GstFlowReturn gst_yoloxparser_transform_ip (GstBaseTransform * trans,
    GstBuffer * buf);

enum
{
  PROP_0,
  PROP_THRESHOLD,
  PROP_MUXER_WIDTH,
  PROP_MUXER_HEIGHT,
};

/* pad templates */

static GstStaticPadTemplate gst_yoloxparser_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY
    );

static GstStaticPadTemplate gst_yoloxparser_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY
    );


/* class initialization */

G_DEFINE_TYPE_WITH_CODE (GstYoloxparser, gst_yoloxparser, GST_TYPE_BASE_TRANSFORM,
  GST_DEBUG_CATEGORY_INIT (gst_yoloxparser_debug_category, "yoloxparser", 0,
  "debug category for yoloxparser element"));

static void
gst_yoloxparser_class_init (GstYoloxparserClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS (klass);

  /* Setting up pads and setting metadata should be moved to
     base_class_init if you intend to subclass this class. */
  gst_element_class_add_static_pad_template (GST_ELEMENT_CLASS(klass),
      &gst_yoloxparser_src_template);
  gst_element_class_add_static_pad_template (GST_ELEMENT_CLASS(klass),
      &gst_yoloxparser_sink_template);

  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS(klass),
      "FIXME Long name", "Generic", "FIXME Description",
      "FIXME <fixme@example.com>");

  gobject_class->set_property = gst_yoloxparser_set_property;
  gobject_class->get_property = gst_yoloxparser_get_property;
  gobject_class->dispose = gst_yoloxparser_dispose;
  gobject_class->finalize = gst_yoloxparser_finalize;
  // base_transform_class->transform_caps = GST_DEBUG_FUNCPTR (gst_yoloxparser_transform_caps);
  // base_transform_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_yoloxparser_fixate_caps);
  // base_transform_class->accept_caps = GST_DEBUG_FUNCPTR (gst_yoloxparser_accept_caps);
  base_transform_class->set_caps = GST_DEBUG_FUNCPTR (gst_yoloxparser_set_caps);
  // base_transform_class->query = GST_DEBUG_FUNCPTR (gst_yoloxparser_query);
  // base_transform_class->decide_allocation = GST_DEBUG_FUNCPTR (gst_yoloxparser_decide_allocation);
  // base_transform_class->filter_meta = GST_DEBUG_FUNCPTR (gst_yoloxparser_filter_meta);
  // base_transform_class->propose_allocation = GST_DEBUG_FUNCPTR (gst_yoloxparser_propose_allocation);
  // base_transform_class->transform_size = GST_DEBUG_FUNCPTR (gst_yoloxparser_transform_size);
  // base_transform_class->get_unit_size = GST_DEBUG_FUNCPTR (gst_yoloxparser_get_unit_size);
  base_transform_class->start = GST_DEBUG_FUNCPTR (gst_yoloxparser_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR (gst_yoloxparser_stop);
  // base_transform_class->sink_event = GST_DEBUG_FUNCPTR (gst_yoloxparser_sink_event);
  // base_transform_class->src_event = GST_DEBUG_FUNCPTR (gst_yoloxparser_src_event);
  // base_transform_class->prepare_output_buffer = GST_DEBUG_FUNCPTR (gst_yoloxparser_prepare_output_buffer);
  // base_transform_class->copy_metadata = GST_DEBUG_FUNCPTR (gst_yoloxparser_copy_metadata);
  // base_transform_class->transform_meta = GST_DEBUG_FUNCPTR (gst_yoloxparser_transform_meta);
  // base_transform_class->before_transform = GST_DEBUG_FUNCPTR (gst_yoloxparser_before_transform);
  // base_transform_class->transform = GST_DEBUG_FUNCPTR (gst_yoloxparser_transform);
  base_transform_class->transform_ip = GST_DEBUG_FUNCPTR (gst_yoloxparser_transform_ip);

  g_object_class_install_property (gobject_class, PROP_THRESHOLD,
      g_param_spec_float ("threshold", "Threshold",
      "Yolox threshold",
      0.1, 1.0, DEFAULT_THRESHOLD,
      (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  
  g_object_class_install_property (gobject_class, PROP_MUXER_WIDTH,
      g_param_spec_int ("muxer-width", "muxer-width",
      "Muxer output width",
      0, 1000000, MUXER_OUTPUT_WIDTH,
      (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  
  g_object_class_install_property (gobject_class, PROP_MUXER_HEIGHT,
      g_param_spec_int ("muxer-height", "muxer-height",
      "Muxer output height",
      0, 1000000, MUXER_OUTPUT_HEIGHT,
      (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
}

static void
gst_yoloxparser_init (GstYoloxparser *yoloxparser)
{
  yoloxparser->threshold = DEFAULT_THRESHOLD;
  yoloxparser->muxer_width = MUXER_OUTPUT_WIDTH;
  yoloxparser->muxer_height = MUXER_OUTPUT_HEIGHT;
}

void
gst_yoloxparser_set_property (GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  GstYoloxparser *yoloxparser = GST_YOLOXPARSER (object);

  GST_DEBUG_OBJECT (yoloxparser, "set_property");

  switch (property_id) {
    case PROP_THRESHOLD:
      yoloxparser->threshold = g_value_get_float(value);
      break;
    case PROP_MUXER_WIDTH:
      yoloxparser->muxer_width = g_value_get_int(value);
      break;
    case PROP_MUXER_HEIGHT:
      yoloxparser->muxer_height = g_value_get_int(value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_yoloxparser_get_property (GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  GstYoloxparser *yoloxparser = GST_YOLOXPARSER (object);

  GST_DEBUG_OBJECT (yoloxparser, "get_property");

  switch (property_id) {
    case PROP_THRESHOLD:
      g_value_set_float (value, yoloxparser->threshold);
      break;
    case PROP_MUXER_WIDTH:
      g_value_set_int (value, yoloxparser->muxer_width);
      break;
    case PROP_MUXER_HEIGHT:
      g_value_set_int (value, yoloxparser->muxer_height);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_yoloxparser_dispose (GObject * object)
{
  GstYoloxparser *yoloxparser = GST_YOLOXPARSER (object);

  GST_DEBUG_OBJECT (yoloxparser, "dispose");

  /* clean up as possible.  may be called multiple times */

  G_OBJECT_CLASS (gst_yoloxparser_parent_class)->dispose (object);
}

void
gst_yoloxparser_finalize (GObject * object)
{
  GstYoloxparser *yoloxparser = GST_YOLOXPARSER (object);

  GST_DEBUG_OBJECT (yoloxparser, "finalize");

  /* clean up object here */

  G_OBJECT_CLASS (gst_yoloxparser_parent_class)->finalize (object);
}

// static GstCaps *
// gst_yoloxparser_transform_caps (GstBaseTransform * trans, GstPadDirection direction,
//     GstCaps * caps, GstCaps * filter)
// {
//   GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);
//   GstCaps *othercaps;

//   GST_DEBUG_OBJECT (yoloxparser, "transform_caps");

//   othercaps = gst_caps_copy (caps);

//   /* Copy other caps and modify as appropriate */
//   /* This works for the simplest cases, where the transform modifies one
//    * or more fields in the caps structure.  It does not work correctly
//    * if passthrough caps are preferred. */
//   if (direction == GST_PAD_SRC) {
//     /* transform caps going upstream */
//   } else {
//     /* transform caps going downstream */
//   }

//   if (filter) {
//     GstCaps *intersect;

//     intersect = gst_caps_intersect (othercaps, filter);
//     gst_caps_unref (othercaps);

//     return intersect;
//   } else {
//     return othercaps;
//   }
// }

// static GstCaps *
// gst_yoloxparser_fixate_caps (GstBaseTransform * trans, GstPadDirection direction,
//     GstCaps * caps, GstCaps * othercaps)
// {
//   GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

//   GST_DEBUG_OBJECT (yoloxparser, "fixate_caps");

//   return NULL;
// }

// static gboolean
// gst_yoloxparser_accept_caps (GstBaseTransform * trans, GstPadDirection direction,
//     GstCaps * caps)
// {
//   GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

//   GST_DEBUG_OBJECT (yoloxparser, "accept_caps");

//   return TRUE;
// }

static gboolean
gst_yoloxparser_set_caps (GstBaseTransform * trans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

  GST_DEBUG_OBJECT (yoloxparser, "set_caps");

  return TRUE;
}

// static gboolean
// gst_yoloxparser_query (GstBaseTransform * trans, GstPadDirection direction,
//     GstQuery * query)
// {
//   GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

//   GST_DEBUG_OBJECT (yoloxparser, "query");

//   return TRUE;
// }

// /* decide allocation query for output buffers */
// static gboolean
// gst_yoloxparser_decide_allocation (GstBaseTransform * trans, GstQuery * query)
// {
//   GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

//   GST_DEBUG_OBJECT (yoloxparser, "decide_allocation");

//   return TRUE;
// }

// static gboolean
// gst_yoloxparser_filter_meta (GstBaseTransform * trans, GstQuery * query, GType api,
//     const GstStructure * params)
// {
//   GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

//   GST_DEBUG_OBJECT (yoloxparser, "filter_meta");

//   return TRUE;
// }

// /* propose allocation query parameters for input buffers */
// static gboolean
// gst_yoloxparser_propose_allocation (GstBaseTransform * trans,
//     GstQuery * decide_query, GstQuery * query)
// {
//   GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

//   GST_DEBUG_OBJECT (yoloxparser, "propose_allocation");

//   return TRUE;
// }

// /* transform size */
// static gboolean
// gst_yoloxparser_transform_size (GstBaseTransform * trans, GstPadDirection direction,
//     GstCaps * caps, gsize size, GstCaps * othercaps, gsize * othersize)
// {
//   GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

//   GST_DEBUG_OBJECT (yoloxparser, "transform_size");

//   return TRUE;
// }

// static gboolean
// gst_yoloxparser_get_unit_size (GstBaseTransform * trans, GstCaps * caps,
//     gsize * size)
// {
//   GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

//   GST_DEBUG_OBJECT (yoloxparser, "get_unit_size");

//   return TRUE;
// }

/* states */
static gboolean
gst_yoloxparser_start (GstBaseTransform * trans)
{
  GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

  GST_DEBUG_OBJECT (yoloxparser, "start");

  return TRUE;
}

static gboolean
gst_yoloxparser_stop (GstBaseTransform * trans)
{
  GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

  GST_DEBUG_OBJECT (yoloxparser, "stop");

  return TRUE;
}

// /* sink and src pad event handlers */
// static gboolean
// gst_yoloxparser_sink_event (GstBaseTransform * trans, GstEvent * event)
// {
//   GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

//   GST_DEBUG_OBJECT (yoloxparser, "sink_event");

//   return GST_BASE_TRANSFORM_CLASS (gst_yoloxparser_parent_class)->sink_event (
//       trans, event);
// }

// static gboolean
// gst_yoloxparser_src_event (GstBaseTransform * trans, GstEvent * event)
// {
//   GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

//   GST_DEBUG_OBJECT (yoloxparser, "src_event");

//   return GST_BASE_TRANSFORM_CLASS (gst_yoloxparser_parent_class)->src_event (
//       trans, event);
// }

// static GstFlowReturn
// gst_yoloxparser_prepare_output_buffer (GstBaseTransform * trans, GstBuffer * input,
//     GstBuffer ** outbuf)
// {
//   GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

//   GST_DEBUG_OBJECT (yoloxparser, "prepare_output_buffer");

//   return GST_FLOW_OK;
// }

// /* metadata */
// static gboolean
// gst_yoloxparser_copy_metadata (GstBaseTransform * trans, GstBuffer * input,
//     GstBuffer * outbuf)
// {
//   GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

//   GST_DEBUG_OBJECT (yoloxparser, "copy_metadata");

//   return TRUE;
// }

// static gboolean
// gst_yoloxparser_transform_meta (GstBaseTransform * trans, GstBuffer * outbuf,
//     GstMeta * meta, GstBuffer * inbuf)
// {
//   GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

//   GST_DEBUG_OBJECT (yoloxparser, "transform_meta");

//   return TRUE;
// }

// static void
// gst_yoloxparser_before_transform (GstBaseTransform * trans, GstBuffer * buffer)
// {
//   GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

//   GST_DEBUG_OBJECT (yoloxparser, "before_transform");

// }

// /* transform */
// static GstFlowReturn
// gst_yoloxparser_transform (GstBaseTransform * trans, GstBuffer * inbuf,
//     GstBuffer * outbuf)
// {
//   GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

//   GST_DEBUG_OBJECT (yoloxparser, "transform");

//   return GST_FLOW_OK;
// }

static GstFlowReturn
gst_yoloxparser_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  GstYoloxparser *yoloxparser = GST_YOLOXPARSER (trans);

  GST_DEBUG_OBJECT (yoloxparser, "transform_ip");

  NvDsInferParseDetectionParams detectionParams;
  detectionParams.perClassThreshold = {yoloxparser->threshold};

  NvDsBatchMeta *batch_meta = 
    gst_buffer_get_nvds_batch_meta (buf);
  /* Iterate each frame metadata in batch */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

    for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
      NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
      if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
        continue;
      
      /* convert to tensor metadata */
      NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
      for (unsigned int i = 0; i < meta->num_output_layers; i++) {
        NvDsInferLayerInfo *info = &meta->output_layers_info[i];
        info->buffer = meta->out_buf_ptrs_host[i];
      }
      /* Parse output tensor and fill detection results into objectList. */
      NvDsInferNetworkInfo& networkInfo = meta->network_info;
      std::vector < NvDsInferLayerInfo >outputLayersInfo (meta->output_layers_info,
        meta->output_layers_info + meta->num_output_layers);
      std::vector < YoloXObjectDetectionInfo > objectList;
      YoloxType yolox_type = NvDsInferParseYoloX(outputLayersInfo, networkInfo,
        detectionParams, objectList);
      
      for (auto & obj:objectList) {
        NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool (batch_meta);
        obj_meta->unique_component_id = meta->unique_id;
        obj_meta->confidence = obj.detectionConfidence;
        obj_meta->object_id = UNTRACKED_OBJECT_ID;
        obj_meta->class_id = 0;

        NvOSD_RectParams & rect_params = obj_meta->rect_params;
        NvOSD_TextParams & text_params = obj_meta->text_params;

        /* Assign bounding box coordinates. */
        rect_params.left = obj.left / networkInfo.width * yoloxparser->muxer_width;
        rect_params.top = obj.top / networkInfo.width * yoloxparser->muxer_width;
        rect_params.width = obj.width / networkInfo.width * yoloxparser->muxer_width;
        rect_params.height = obj.height / networkInfo.width * yoloxparser->muxer_width;

        if (yolox_type == YoloX_Fisheye) {
          FisheyeData *box_meta = (FisheyeData *) g_malloc0 (sizeof (FisheyeData));
          NvOSD_RectParams & full_bbox = box_meta->full_bbox;
          NvOSD_RectParams & foot_bbox = box_meta->foot_bbox;
          RotateBox & rotate_bbox = box_meta->rotate_bbox;

          // update rotate_bbox
          rotate_bbox.tl = Point(obj.pts[2].x / networkInfo.width * yoloxparser->muxer_width,
                                 obj.pts[2].y / networkInfo.width * yoloxparser->muxer_width);
          rotate_bbox.tr = Point(obj.pts[1].x / networkInfo.width * yoloxparser->muxer_width,
                                 obj.pts[1].y / networkInfo.width * yoloxparser->muxer_width);
          rotate_bbox.br = Point(obj.pts[0].x / networkInfo.width * yoloxparser->muxer_width,
                                 obj.pts[0].y / networkInfo.width * yoloxparser->muxer_width);
          rotate_bbox.bl = Point(obj.pts[3].x / networkInfo.width * yoloxparser->muxer_width,
                                 obj.pts[3].y / networkInfo.width * yoloxparser->muxer_width);

          // update foot_bbox
          foot_bbox = rect_params;

          // update full_bbox
          full_bbox.left = std::min({obj.pts[0].x, obj.pts[1].x, obj.pts[2].x, obj.pts[3].x});
          full_bbox.top = std::min({obj.pts[0].y, obj.pts[1].y, obj.pts[2].y, obj.pts[3].y});
          full_bbox.width = std::max({obj.pts[0].x, obj.pts[1].x, obj.pts[2].x, obj.pts[3].x}) - full_bbox.left;
          full_bbox.height = std::max({obj.pts[0].y, obj.pts[1].y, obj.pts[2].y, obj.pts[3].y}) - full_bbox.top;
          full_bbox.left = full_bbox.left / networkInfo.width * yoloxparser->muxer_width;
          full_bbox.top = full_bbox.top / networkInfo.width * yoloxparser->muxer_width;
          full_bbox.width = full_bbox.width / networkInfo.width * yoloxparser->muxer_width;
          full_bbox.height = full_bbox.height / networkInfo.width * yoloxparser->muxer_width;
          /* verify bbox */
          full_bbox.left = full_bbox.left * (full_bbox.left>=0.);
          full_bbox.top = full_bbox.top * (full_bbox.top>=0.);
          full_bbox.width -= (full_bbox.left + full_bbox.width > yoloxparser->muxer_width) * (full_bbox.left + full_bbox.width - yoloxparser->muxer_width);
          full_bbox.height -= (full_bbox.top + full_bbox.height > yoloxparser->muxer_height) * (full_bbox.top + full_bbox.height - yoloxparser->muxer_height);

          NvDsUserMeta *user_box_meta = nvds_acquire_user_meta_from_pool (batch_meta);
          user_box_meta->user_meta_data = (void *) box_meta;
          user_box_meta->base_meta.meta_type = NVDS_GST_CUSTOM_META;
          user_box_meta->base_meta.copy_func = (NvDsMetaCopyFunc) box_meta_copy_func;
          user_box_meta->base_meta.release_func = (NvDsMetaReleaseFunc) box_meta_free_func;
          nvds_add_user_meta_to_obj(obj_meta, user_box_meta);
        }

        /* verify bbox */
        rect_params.left = rect_params.left * (rect_params.left>=0.);
        rect_params.top = rect_params.top * (rect_params.top>=0.);
        rect_params.width -= (rect_params.left + rect_params.width > yoloxparser->muxer_width) * (rect_params.left + rect_params.width - yoloxparser->muxer_width);
        rect_params.height -= (rect_params.top + rect_params.height > yoloxparser->muxer_height) * (rect_params.top + rect_params.height - yoloxparser->muxer_height);

        /* Border of width 3. */
        rect_params.border_width = 3;
        rect_params.has_bg_color = 0;
        rect_params.border_color = (NvOSD_ColorParams) {1, 0, 0, 1};

        /* display_text requires heap allocated memory. */
        text_params.display_text = g_strdup("person");
        /* Display text above the left top corner of the object. */
        text_params.x_offset = rect_params.left;
        text_params.y_offset = rect_params.top - 10;
        /* Set black background for the text. */
        text_params.set_bg_clr = 1;
        text_params.text_bg_clr = (NvOSD_ColorParams) {0, 0, 0, 1};
        /* Font face, size and color. */
        text_params.font_params.font_name = (gchar *) "Serif";
        text_params.font_params.font_size = 11;
        text_params.font_params.font_color = (NvOSD_ColorParams) {1, 1, 1, 1};
        nvds_add_obj_meta_to_frame (frame_meta, obj_meta, NULL);
      }
    }
  }
  return GST_FLOW_OK;
}

static gboolean
plugin_init (GstPlugin * plugin)
{

  /* FIXME Remember to set the rank if it's an element that is meant
     to be autoplugged by decodebin. */
  return gst_element_register (plugin, "yoloxparser", GST_RANK_NONE,
      GST_TYPE_YOLOXPARSER);
}

/* FIXME: these are normally defined by the GStreamer build system.
   If you are creating an element to be included in gst-plugins-*,
   remove these, as they're always defined.  Otherwise, edit as
   appropriate for your external plugin package. */
#ifndef VERSION
#define VERSION "0.0.FIXME"
#endif
#ifndef PACKAGE
#define PACKAGE "FIXME_package"
#endif
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "FIXME_package_name"
#endif
#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "http://FIXME.org/"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    yoloxparser,
    "FIXME plugin description",
    plugin_init, VERSION, "LGPL", PACKAGE_NAME, GST_PACKAGE_ORIGIN)

