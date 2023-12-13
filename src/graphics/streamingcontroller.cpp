#include "streamingcontroller.hpp"

#include "../config/graphics.hpp"
#include <stdexcept>
#include <iostream>


StremmingController::StremmingController(const std::string& host, int port):
	host(host), port(port), pixels(windowHeight * windowWidth * 4), timestamp(0)
{
	GstStateChangeReturn ret;

	/* init GStreamer */
	gst_init (NULL, NULL);

	/* setup pipeline */
	appsrc = gst_element_factory_make ("appsrc", "source");
	x264enc = gst_element_factory_make("x264enc", "encoder");
	rtph264pay = gst_element_factory_make("rtph264pay", "payload");
	udpsink = gst_element_factory_make("udpsink", "sink");
	convert = gst_element_factory_make("videoconvert", "converter");

	pipeline = gst_pipeline_new ("pipeline");

	if (!pipeline || !appsrc || !x264enc || !rtph264pay || !udpsink || !convert) {
		throw std::runtime_error("Error while creating pipeline elements");
	}

	/* setup */
  	g_object_set(G_OBJECT(appsrc), "caps",
  		gst_caps_new_simple ("video/x-raw",
             "format", G_TYPE_STRING, "RGBA",
				     "width", G_TYPE_INT, windowWidth,
				     "height", G_TYPE_INT, windowHeight,
				     "framerate", GST_TYPE_FRACTION, 0, 1,
				     NULL), NULL);

	g_object_set(G_OBJECT(udpsink),
		"host", host.data(),
		"port", port,
		NULL);

	gst_bin_add_many (GST_BIN (pipeline), appsrc, convert, x264enc, rtph264pay, udpsink, NULL);

	if (gst_element_link_many (appsrc, convert, x264enc, rtph264pay, udpsink, NULL) != TRUE) {
		gst_object_unref(pipeline);
		throw std::runtime_error("Cannot link objects to pipeline");
	}

	/* setup appsrc */
	g_object_set (G_OBJECT (appsrc),
		"stream-type", 0, // GST_APP_STREAM_TYPE_STREAM
		"format", GST_FORMAT_TIME,
		"is-live", TRUE,
		"block", TRUE,
		NULL);

	/* play */
	ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
	if (ret == GST_STATE_CHANGE_FAILURE) {
        gst_object_unref(pipeline);
        throw std::runtime_error("Cannot set pipeline to playing state");
    }
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

	buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_LAST, (gpointer)(pixels.data()), pixels.size(), 0, pixels.size(), NULL, NULL );

	GST_BUFFER_PTS (buffer) = timestamp;
	GST_BUFFER_DURATION (buffer) = gst_util_uint64_scale_int(1, GST_SECOND, 30);

	timestamp += GST_BUFFER_DURATION (buffer);

	if (gst_app_src_push_buffer((GstAppSrc*)appsrc, buffer) != GST_FLOW_OK) {
		throw std::runtime_error("Error while pushing data to buffer");
	}
}
