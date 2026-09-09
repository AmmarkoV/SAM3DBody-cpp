/** @file glx3.h
 *  @brief  X Server bindings to create an OpenGL 3.x+ context and start rendering to a window
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef GLX3_H_INCLUDED
#define GLX3_H_INCLUDED

int disableVSync();

/**
* @brief Set the title to give the window the next time it is created.
* Must be called before start_glx3_stuff(); passing NULL/empty resets to the
* built-in default ("SAM3DBody-cpp OpenGL3.x+ Visualization").
* @ingroup X11
*/
void glx3_set_window_title(const char *title);

/**
* @brief create a glx window that can serve OpenGL draw requests
* @ingroup X11
* @param width , The width of the window in pixels
* @param height, The height of the window in pixels
* @param viewWindow, Setting this value to zero will make the "window" invisible
* @param argc, Number of input arguments from main
* @param argv, Pointer to an array of strings from main
* @retval 1=Success , 0=Failure
*/
int start_glx3_stuff(int WIDTH,int HEIGHT,int viewWindow,int argc,const char **argv);


int stop_glx3_stuff();

/**
* @brief After drawing everything on our OpenGL window this call swaps buffers and outputs
* @ingroup X11
* @retval 1=Success , 0=Failure
*/
int glx3_endRedraw();


/**
* @brief Pump pending X events.
*
* Returns 1 while the GL surface is still alive and the application should
* keep its render loop going.  Returns 0 once a close has been requested —
* either by the user clicking the window-manager's close button (which
* delivers a `WM_DELETE_WINDOW` ClientMessage we now honour), pressing the
* Escape key, or another call site invoking glx3_request_close().
*
* Render loops should be `while (glx3_checkEvents()) { ... }` so that the
* close signal terminates the loop cleanly and the binary can run its
* normal post-loop teardown (BVH close, mp4 encode, etc.) instead of being
* hard-killed.
*
* @ingroup X11
* @retval 1=keep running, 0=close requested
*/
int glx3_checkEvents();

/**
* @brief Returns non-zero once a clean shutdown has been requested.
* Independent of glx3_checkEvents() so non-X-event-driven loops (e.g. an
* OOM handler, a timer, the BVH writer detecting end-of-video) can poll
* it as well.
*/
int glx3_should_close();

/**
* @brief Programmatically request a clean shutdown.  Subsequent
* glx3_checkEvents() / glx3_should_close() calls will return 0 / non-zero.
*/
void glx3_request_close();

#endif // GLX2_H_INCLUDED
