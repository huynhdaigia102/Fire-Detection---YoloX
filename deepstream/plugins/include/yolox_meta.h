#pragma once

#include "nvll_osd_struct.h"


struct Point {
  float x, y;
  Point(const float& px = 0, const float& py = 0) : x(px), y(py) {};
};


typedef struct _RotateBox{
  Point tl;
  Point tr;
  Point br;
  Point bl;
} RotateBox;

typedef struct _FisheyeData{
  NvOSD_RectParams full_bbox;
  NvOSD_RectParams foot_bbox;
  RotateBox rotate_bbox;
} FisheyeData;


typedef enum {
  YoloX_None,
  YoloX_Normal,
  YoloX_Fisheye,
} YoloxType;

typedef struct
{
  /** Holds the ID of the class to which the object belongs. */
  unsigned int classId;

  /** Holds the horizontal offset of the bounding box shape for the object. */
  float left;
  /** Holds the vertical offset of the object's bounding box. */
  float top;
  /** Holds the width of the object's bounding box. */
  float width;
  /** Holds the height of the object's bounding box. */
  float height;

  /** Holds the object detection confidence level; must in the range
   [0.0,1.0]. */
  float detectionConfidence;

  /** Holds the vertices of the object's bounding box. */
  Point pts[4];

  /** Holds the foot of the object's bounding box. */
  Point foot_pt;

} YoloXObjectDetectionInfo;
