/** @file win32_gl.h
 *  @brief  Win32 bindings to create an OpenGL 3.x+ context and start rendering to a window
 */

#ifndef WIN32_GL_H_INCLUDED
#define WIN32_GL_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

int disableVSync();

/**
* @brief create a window that can serve OpenGL draw requests
* @param WIDTH, The width of the window in pixels
* @param HEIGHT, The height of the window in pixels
* @param viewWindow, Setting this value to zero will make the "window" invisible
* @param argc, Number of input arguments from main
* @param argv, Pointer to an array of strings from main
* @retval 1=Success , 0=Failure
*/
int start_glx3_stuff(int WIDTH, int HEIGHT, int viewWindow, int argc, const char **argv);

int stop_glx3_stuff();

/**
* @brief After drawing everything on our OpenGL window this call swaps buffers and outputs
* @retval 1=Success , 0=Failure
*/
int glx3_endRedraw();

/**
* @brief Pump pending Windows messages.
*
* Returns 1 while the GL surface is still alive. Returns 0 once a close has been requested.
* @retval 1=keep running, 0=close requested
*/
int glx3_checkEvents();

/**
* @brief Returns non-zero once a clean shutdown has been requested.
*/
int glx3_should_close();

/**
* @brief Programmatically request a clean shutdown.
*/
void glx3_request_close();

#ifdef __cplusplus
}
#endif

#endif
