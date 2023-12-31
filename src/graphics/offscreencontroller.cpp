#include "offscreencontroller.hpp"

#include "../config/graphics.hpp"

#include <sstream>
#include <iostream>


static const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
};

static const EGLint pbufferAttribs[] = {
        EGL_WIDTH, windowWidth,
        EGL_HEIGHT, windowHeight,
        EGL_NONE,
  };


OffscreenController::OffscreenController()
{
    EGLint major, minor;
    EGLint numConfigs;
    EGLConfig eglCfg;

    // 1. Initialize EGL
    eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    assertEGLError("eglGetDisplay");

    eglInitialize(eglDpy, &major, &minor);
    assertEGLError("eglInitialize");

    // 2. Select an appropriate configuration
    eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);
    assertEGLError("eglChooseConfig");

    // 3. Create a surface
    eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);

    // 4. Bind the API
    eglBindAPI(EGL_OPENGL_API);
    assertEGLError("eglBindAPI");

    // 5. Create a context and make it current
    eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, 
                                        NULL);
    assertEGLError("eglCreateContext");

    eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);
    assertEGLError("eglMakeCurrent");   
}

OffscreenController::~OffscreenController()
{
    eglDestroyContext(eglDpy, eglCtx);
    assertEGLError("eglDestroyContext");

    eglTerminate(eglDpy);
    assertEGLError("eglTerminate");
}

void OffscreenController::assertEGLError(const std::string & msg)
{
    EGLint error = eglGetError();

	if (error != EGL_SUCCESS) {
		std::stringstream s;
		s << "EGL error 0x" << std::hex << error << " at " << msg;
		throw std::runtime_error(s.str());
	}
}
