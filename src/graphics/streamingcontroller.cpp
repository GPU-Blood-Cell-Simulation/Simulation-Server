#include "streamingcontroller.hpp"

#include "../config/graphics.hpp"
#include <stdexcept>


StremmingController::StremmingController():
	pixels(windowHeight * windowWidth * 4), timestamp(0)
{
		/* init GStreamer */
	gst_init (NULL, NULL);

	/* setup pipeline */
	pipeline = gst_pipeline_new ("pipeline");
	appsrc = gst_element_factory_make ("appsrc", "source");
	conv = gst_element_factory_make ("videoconvert", "conv");
	videosink = gst_element_factory_make ("xvimagesink", "videosink");

	/* setup */
  	g_object_set (G_OBJECT (appsrc), "caps",
  		gst_caps_new_simple ("video/x-raw",
				     /*"format", G_TYPE_STRING, "RGB16",*/
             "format", G_TYPE_STRING, "RGBA",
				     "width", G_TYPE_INT, windowWidth,
				     "height", G_TYPE_INT, windowHeight,
				     "framerate", GST_TYPE_FRACTION, 0, 1,
				     NULL), NULL);

	gst_bin_add_many (GST_BIN (pipeline), appsrc, conv, videosink, NULL);
	gst_element_link_many (appsrc, conv, videosink, NULL);

	/* setup appsrc */
	g_object_set (G_OBJECT (appsrc),
		"stream-type", 0, // GST_APP_STREAM_TYPE_STREAM
		"format", GST_FORMAT_TIME,
		"is-live", TRUE,
		"block", TRUE,
		NULL);

	/* play */
	gst_element_set_state (pipeline, GST_STATE_PLAYING);
}


StremmingController::~StremmingController()
{
	gst_element_set_state (pipeline, GST_STATE_NULL);
	gst_object_unref (GST_OBJECT (pipeline));
}


void StremmingController::SendFrame()
{
  GstBuffer *buffer;

  glReadPixels(0, 0, windowWidth, windowHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

  buffer = gst_buffer_new_wrapped_full( GST_MEMORY_FLAG_LAST, (gpointer)(pixels.data()), pixels.size(), 0, pixels.size(), NULL, NULL );

  GST_BUFFER_PTS (buffer) = timestamp;
  GST_BUFFER_DURATION (buffer) = gst_util_uint64_scale_int (1, GST_SECOND, 30);

  timestamp += GST_BUFFER_DURATION (buffer);

  if (gst_app_src_push_buffer((GstAppSrc*)appsrc, buffer) != GST_FLOW_OK) {
    throw std::runtime_error("Error while pushing data to buffer");
  }
}
