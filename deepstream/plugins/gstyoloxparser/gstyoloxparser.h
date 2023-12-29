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
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#ifndef _GST_YOLOXPARSER_H_
#define _GST_YOLOXPARSER_H_

#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_YOLOXPARSER   (gst_yoloxparser_get_type())
#define GST_YOLOXPARSER(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_YOLOXPARSER,GstYoloxparser))
#define GST_YOLOXPARSER_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_YOLOXPARSER,GstYoloxparserClass))
#define GST_IS_YOLOXPARSER(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_YOLOXPARSER))
#define GST_IS_YOLOXPARSER_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_YOLOXPARSER))

#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080
#define DEFAULT_THRESHOLD 0.1

typedef struct _GstYoloxparser GstYoloxparser;
typedef struct _GstYoloxparserClass GstYoloxparserClass;

struct _GstYoloxparser
{
  GstBaseTransform base_yoloxparser;
  gfloat threshold;
  gint muxer_width;
  gint muxer_height;
};

struct _GstYoloxparserClass
{
  GstBaseTransformClass base_yoloxparser_class;
};

GType gst_yoloxparser_get_type (void);

G_END_DECLS

#endif
