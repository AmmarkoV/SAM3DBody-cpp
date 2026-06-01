//From : https://www.khronos.org/opengl/wiki/Tutorial:_OpenGL_3.0_Context_Creation_(GLX)
//Compile using :  gcc -o gl3 glx3.c -lGL -lX11
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>          /* sig_atomic_t */
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>



#include <GL/gl.h>
#include <GL/glx.h>

#include "glx3.h"

extern int handleUserInput(int key, int pressed, int x, int y);
extern int windowSizeUpdated(int width, int height);

GLXDrawable whatToSwap=0;
/* Set to 1 by the Pbuffer (offscreen) code path in start_glx3_stuff(); stays 0
 * in the windowed path.  stop_glx3_stuff() uses this to skip XDestroyWindow /
 * XFreeColormap, which fail with BadWindow in Pbuffer mode because `win` and
 * `cmap` were never initialised — and the failure happens at the very END of
 * a successful run, after every frame has been written, so it would propagate
 * a non-zero exit code that scripts/video.sh interprets as "renderer died,
 * abort the encode".  Concretely: matrix.mp4 saved all 2182 JPEGs, then this
 * cleanup raised an X BadWindow error, the process exited 1, and the script
 * discarded the frames. */
static int glx3_is_pbuffer = 0;
GLboolean  doubleBufferGLX3 = GL_TRUE;
//static int dblBuf[]  = {GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None};

Display   *display;
Window     win;
GLXContext ctx = 0;
Colormap cmap;

/* Clean-shutdown plumbing.  WM_DELETE_WINDOW is the X11 protocol for "user
 * clicked the close button" — without registering for it, the WM kills the
 * X connection on close, which historically meant our process was killed
 * mid-frame (truncated mp4 output, half-written BVH, no encoder hand-off).
 * We register, catch the corresponding ClientMessage in glx3_checkEvents(),
 * and turn it into a clean shutdown via the close flag below.
 *
 * The flag is also exposed via glx3_request_close() so non-X paths
 * (Escape key, end-of-video, BVH writer aborts, future signal handlers)
 * can ask for the same clean exit. */
static Atom wm_delete_window = 0;
static volatile sig_atomic_t close_requested = 0;


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */


#define GLX_CONTEXT_MAJOR_VERSION_ARB       0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB       0x2092
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);



void fatalErrorGLX3(char *message)
{
    fprintf(stderr, RED "fatal Error: %s\n" NORMAL, message);
    //exit(1);
}


// Helper to check for extension string presence.  Adapted from:
//   http://www.opengl.org/resources/features/OGLextensions/
static int isExtensionSupported(const char *extList, const char *extension)
{
    const char *start;
    const char *where, *terminator;

    /* Extension names should not have spaces. */
    where = strchr(extension, ' ');
    if (where || *extension == '\0')
        return 0;

    /* It takes a bit of care to be fool-proof about parsing the
       OpenGL extensions string. Don't be fooled by sub-strings,
       etc. */
    for (start=extList;;)
    {
        where = strstr(start, extension);

        if (!where)
            break;

        terminator = where + strlen(extension);

        if ( where == start || *(where - 1) == ' ' )
            if ( *terminator == ' ' || *terminator == '\0' )
                return 1;

        start = terminator;
    }

    return 0;
}

static int ctxErrorOccurred = 0;
static int ctxErrorHandler( Display *dpy, XErrorEvent *ev )
{
    ctxErrorOccurred = 1;
    return 0;
}




static void run_getenv (const char * name)
{
    char * value = getenv (name);
    if (! value)
    {
        printf ("'%s' is not set.\n", name);
    }
    else
    {
        printf ("%s = %s\n", name, value);
    }
}


int disableVSync() //This needs to be done before initialization of GLX3 stuff..
{
    //NVIDIA VSYNC ENVIRONMENT FLAG
    run_getenv("__GL_SYNC_TO_VBLANK");
    setenv ("__GL_SYNC_TO_VBLANK", "0", 0);
    run_getenv("__GL_SYNC_TO_VBLANK");

    //INTEL VSYNC ENVIRONMENT FLAG
    run_getenv("vblank_mode");
    setenv ("vblank_mode", "0", 0);
    run_getenv("vblank_mode");
    return 1;
}


int start_glx3_stuffWindowed(int WIDTH,int HEIGHT,int argc,const char **argv)
{
    fprintf(stderr,"start_glx3_stuffWindowed\n");
    doubleBufferGLX3 = GL_TRUE;

    display = XOpenDisplay(NULL);

    if (!display)
    {
        printf("Failed to open X display\n");
        return 0;
    }

    // Get a matching FB config
    static int visual_attribs[] =
    {
        GLX_X_RENDERABLE, True,
        GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
        GLX_RENDER_TYPE, GLX_RGBA_BIT,
        GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
        GLX_RED_SIZE, 8,
        GLX_GREEN_SIZE, 8,
        GLX_BLUE_SIZE, 8,
        GLX_ALPHA_SIZE, 8,
        GLX_DEPTH_SIZE, 24,
        GLX_STENCIL_SIZE, 8,
        GLX_DOUBLEBUFFER, True,
        //GLX_SAMPLE_BUFFERS  , 1,
        //GLX_SAMPLES         , 4,
        None
    };

    int glx_major, glx_minor;

    // FBConfigs were added in GLX version 1.3.
    if ( !glXQueryVersion( display, &glx_major, &glx_minor ) ||
            ( ( glx_major == 1 ) && ( glx_minor < 3 ) ) || ( glx_major < 1 ) )
    {
        printf("Invalid GLX version");
        return 0;
    }

    printf( "Getting matching framebuffer configs\n" );
    int fbcount;
    GLXFBConfig* fbc = glXChooseFBConfig(display, DefaultScreen(display), visual_attribs, &fbcount);
    if (!fbc)
    {
        printf( "Failed to retrieve a framebuffer config\n" );
        return 0;
    }
    printf( "Found %d matching FB configs.\n", fbcount );

    // Pick the FB config/visual with the most samples per pixel
    printf( "Getting XVisualInfos\n" );
    int best_fbc = -1, worst_fbc = -1, best_num_samp = -1, worst_num_samp = 999;

    int i;
    for (i=0; i<fbcount; ++i)
    {
        XVisualInfo *vi = glXGetVisualFromFBConfig( display, fbc[i] );
        if ( vi )
        {
            int samp_buf, samples;
            glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLE_BUFFERS, &samp_buf );
            glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLES, &samples  );

            printf( "  Matching fbconfig %d, visual ID 0x%2x: SAMPLE_BUFFERS = %d, SAMPLES = %d\n", i, (unsigned int) vi -> visualid, samp_buf, samples );

            if ( best_fbc < 0 || (samp_buf && samples > best_num_samp) )
                best_fbc = i, best_num_samp = samples;
            if ( worst_fbc < 0 || !samp_buf || samples < worst_num_samp )
                worst_fbc = i, worst_num_samp = samples;
        }
        XFree( vi );
    }

    GLXFBConfig bestFbc = fbc[ best_fbc ];

    // Be sure to free the FBConfig list allocated by glXChooseFBConfig()
    XFree( fbc );

    // Get a visual
    XVisualInfo *vi = glXGetVisualFromFBConfig( display, bestFbc );
    printf( "Chosen visual ID = 0x%x\n",(unsigned int)  vi->visualid );

    printf( "Creating colormap\n" );
    XSetWindowAttributes swa;
    swa.colormap = cmap = XCreateColormap( display,
                                           RootWindow( display, vi->screen ),
                                           vi->visual, AllocNone );
    swa.background_pixmap = None ;
    swa.border_pixel      = 0;
    //swa.event_mask        = StructureNotifyMask;
    swa.event_mask        =  KeyPressMask    | ExposureMask  | ButtonPressMask | StructureNotifyMask;

    printf( "Creating window\n" );
    win = XCreateWindow( display, RootWindow( display, vi->screen ),
                         0, 0, WIDTH /*Width*/, HEIGHT/*Height*/, 0, vi->depth, InputOutput,
                         vi->visual,
                         CWBorderPixel|CWColormap|CWEventMask, &swa );
    if ( !win )
    {
        printf( "Failed to create window.\n" );
        return 0;
    }

    whatToSwap = win;

    // Done with the visual info data
    XFree( vi );

    XStoreName( display, win, "SAM3DBody-cpp OpenGL3.x+ Visualization" );

    /* Subscribe to WM_DELETE_WINDOW so the close button delivers a
     * ClientMessage we can intercept instead of having the WM cut our X
     * connection (which would terminate the process at whatever the next
     * X call happened to be — historically: somewhere in the middle of a
     * frame, taking the truncated-mp4 issue with it). */
    wm_delete_window = XInternAtom( display, "WM_DELETE_WINDOW", False );
    if ( wm_delete_window != None )
        XSetWMProtocols( display, win, &wm_delete_window, 1 );

    printf( "Mapping window\n" );
    XMapWindow( display, win );

    // Get the default screen's GLX extension list
    const char *glxExts = glXQueryExtensionsString( display,
                          DefaultScreen( display ) );

    // NOTE: It is not necessary to create or make current to a context before
    // calling glXGetProcAddressARB
    glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
    glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)
                                 glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB" );


    // Install an X error handler so the application won't exit if GL 3.0
    // context allocation fails.
    //
    // Note this error handler is global.  All display connections in all threads
    // of a process use the same error handler, so be sure to guard against other
    // threads issuing X commands while this code is running.
    ctxErrorOccurred = 0;
    int (*oldHandler)(Display*, XErrorEvent*) =
        XSetErrorHandler(&ctxErrorHandler);

    // Check for the GLX_ARB_create_context extension string and the function.
    // If either is not present, use GLX 1.3 context creation method.
    if ( !isExtensionSupported( glxExts, "GLX_ARB_create_context" ) ||
            !glXCreateContextAttribsARB )
    {
        printf( "glXCreateContextAttribsARB() not found"
                " ... using old-style GLX context\n" );
        ctx = glXCreateNewContext( display, bestFbc, GLX_RGBA_TYPE, 0, True );
    }
    // If it does, try to get a GL 3.0 context!
    else
    {
        int context_attribs[] =
        {
            GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
            GLX_CONTEXT_MINOR_VERSION_ARB, 0,
            //GLX_CONTEXT_FLAGS_ARB        , GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
            None
        };

        printf( "Creating context\n" );
        ctx = glXCreateContextAttribsARB( display, bestFbc, 0,
                                          True, context_attribs );

        // Sync to ensure any errors generated are processed.
        XSync( display, False );
        if ( !ctxErrorOccurred && ctx )
            printf( "Created GL 3.0 context\n" );
        else
        {
            // Couldn't create GL 3.0 context.  Fall back to old-style 2.x context.
            // When a context version below 3.0 is requested, implementations will
            // return the newest context version compatible with OpenGL versions less
            // than version 3.0.
            // GLX_CONTEXT_MAJOR_VERSION_ARB = 1
            context_attribs[1] = 1;
            // GLX_CONTEXT_MINOR_VERSION_ARB = 0
            context_attribs[3] = 0;

            ctxErrorOccurred = 0;

            printf( "Failed to create GL 3.0 context"
                    " ... using old-style GLX context\n" );
            ctx = glXCreateContextAttribsARB( display, bestFbc, 0,
                                              True, context_attribs );
        }
    }

    // Sync to ensure any errors generated are processed.
    XSync( display, False );

    // Restore the original error handler
    XSetErrorHandler( oldHandler );

    if ( ctxErrorOccurred || !ctx )
    {
        printf( "Failed to create an OpenGL context\n" );
        return 0;
    }

    // Verifying that context is a direct context
    if ( ! glXIsDirect ( display, ctx ) )
    {
        printf( "Indirect GLX rendering context obtained\n" );
    }
    else
    {
        printf( "Direct GLX rendering context obtained\n" );
    }

    printf( "Making context current\n" );
    glXMakeCurrent( display, win, ctx );




// glClearColor( 0, 0.0, 0, 1 );
// glClear( GL_COLOR_BUFFER_BIT );
// glXSwapBuffers ( display, win );

    printf( "GLX3.0 context ready..\n" );
    return 1;
}






/* When the offscreen Pbuffer path is unavailable — either no FBConfig exposes
 * GLX_PBUFFER_BIT, or glXCreatePbuffer() fails — fall back to a normal on-screen
 * window instead of aborting via fatalErrorGLX3().
 *
 * This matters because scripts/video.sh --save forces --headless, which asks for
 * a Pbuffer.  Drivers/servers that don't support Pbuffers (some Mesa configs,
 * remote/VirtualGL setups, etc.) used to take down the whole save pipeline with
 * "P-Buffers not supported." → 0 frames written → encode aborted.  A visible
 * window renders the same frames, so --save still works there.
 *
 * Two bits of state must be corrected on the way out:
 *   - glx3_is_pbuffer is reset to 0 so stop_glx3_stuff() uses the windowed
 *     cleanup path (XDestroyWindow / XFreeColormap) rather than the Pbuffer one.
 *   - the display opened by the Pbuffer path is closed, because
 *     start_glx3_stuffWindowed() opens (and reassigns the global) its own. */
static int glx3_pbuffer_fallback_to_window(const char *why,int WIDTH,int HEIGHT,int argc,const char **argv)
{
    fprintf(stderr,"Pbuffer unavailable (%s) — falling back to an on-screen window.\n",why);
    glx3_is_pbuffer = 0;
    if (display) { XCloseDisplay(display); display = 0; }
    return start_glx3_stuffWindowed(WIDTH,HEIGHT,argc,argv);
}

int start_glx3_stuff(int WIDTH,int HEIGHT,int viewWindow,int argc,const char **argv)
{
    if (viewWindow==0)
    {
        fprintf(stderr,"start_glx3_stuff with no window..\n");
        glx3_is_pbuffer = 1;
        display = XOpenDisplay(NULL);
        if (!display)
        {
            printf("Failed to open X display\n");
            return 0;
        }


        int dummy;
        if(!glXQueryExtension(display, &dummy, &dummy))
        {
            fatalErrorGLX3("X server has no OpenGL GLX extension");
            return 0;
        }


        fprintf(stderr,"Will try not to display a window..\n");
        int w=WIDTH, h=HEIGHT;



        int pbuff_visualAttribs[] =
        {
            GLX_RENDER_TYPE, GLX_RGBA_BIT,
            GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
            GLX_MAX_PBUFFER_WIDTH, w,
            GLX_MAX_PBUFFER_HEIGHT, h,
            GLX_DOUBLEBUFFER, True,
            GLX_X_RENDERABLE, GL_TRUE,
            GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
            GLX_RED_SIZE, 8,
            GLX_GREEN_SIZE, 8,
            GLX_BLUE_SIZE, 8,
            GLX_ALPHA_SIZE, 8,
            GLX_DEPTH_SIZE, 24,
            None
        };
        /*
              int visualAttribs[]={
                                   GLX_RENDER_TYPE, GLX_RGBA_BIT,
                                   GLX_MAX_PBUFFER_WIDTH, w,
                                   GLX_MAX_PBUFFER_HEIGHT, h,
                                   GLX_RED_SIZE, 4,
                                   GLX_GREEN_SIZE, 4,
                                   GLX_BLUE_SIZE, 4,
                                   GLX_DRAWABLE_TYPE,GLX_PBUFFER_BIT,
                                   GLX_DEPTH_SIZE, 24,
                                   GLX_DOUBLEBUFFER    , True,
                                   None
                                   };*/

        int numberOfFramebufferConfigurations = 0;
        fprintf(stderr,"glXChooseFBConfig\n");
        GLXFBConfig* fbConfigs = glXChooseFBConfig( display, DefaultScreen(display), pbuff_visualAttribs, &numberOfFramebufferConfigurations );
        if ( (fbConfigs == NULL) || (numberOfFramebufferConfigurations <= 0) )
        {
            return glx3_pbuffer_fallback_to_window("no PBUFFER FBConfig",WIDTH,HEIGHT,argc,argv);
        }

        printf( "Found %d matching FB configs.\n", numberOfFramebufferConfigurations );
        //XVisualInfo *vi = glXGetVisualFromFBConfig( display, fbConfigs[0] );
        //printf( "Chosen visual ID = 0x%x\n", vi->visualid );

        int pbufferAttribs[]=
        {
            GLX_PBUFFER_WIDTH, w,
            GLX_PBUFFER_HEIGHT, h,
            GLX_PRESERVED_CONTENTS, True,
            GLX_NONE
        };

        fprintf(stderr,"glXCreatePbuffer\n");
        GLXPbuffer pbuffer = glXCreatePbuffer( display,fbConfigs[0], pbufferAttribs );
        if (pbuffer==0)
        {
            XFree(fbConfigs);
            return glx3_pbuffer_fallback_to_window("glXCreatePbuffer failed",WIDTH,HEIGHT,argc,argv);
        }



        // Get the default screen's GLX extension list
        const char *glxExts = glXQueryExtensionsString( display, DefaultScreen( display ) );

        // NOTE: It is not necessary to create or make current to a context before
        // calling glXGetProcAddressARB
        glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
        glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc) glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB" );


        // Install an X error handler so the application won't exit if GL 3.0
        // context allocation fails.
        //
        // Note this error handler is global.  All display connections in all threads
        // of a process use the same error handler, so be sure to guard against other
        // threads issuing X commands while this code is running.
        ctxErrorOccurred = 0;
        int (*oldHandler)(Display*, XErrorEvent*) = XSetErrorHandler(&ctxErrorHandler);


        if ( !isExtensionSupported( glxExts, "GLX_ARB_create_context" ) || !glXCreateContextAttribsARB )
        {
            printf( "glXCreateContextAttribsARB() not found ... using old-style GLX context\n" );
            ctx = glXCreateNewContext(
                      display,
                      fbConfigs[0],
                      GLX_RGBA_TYPE,
                      NULL,
                      GL_TRUE
                  );
        }
        else
            // If it does, try to get a GL 3.0 context!
        {
            int context_attribs[] =
            {
                GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
                GLX_CONTEXT_MINOR_VERSION_ARB, 0,
                //GLX_CONTEXT_FLAGS_ARB        , GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
                None
            };

            printf( "Creating context\n" );
            ctx = glXCreateContextAttribsARB( display, fbConfigs[0], 0, True, context_attribs );

            // Sync to ensure any errors generated are processed.
            XSync( display, False );
            if ( !ctxErrorOccurred && ctx )
            {
                printf( "Created GL 3.0 context\n" );
            }
            else
            {
                // Couldn't create GL 3.0 context.  Fall back to old-style 2.x context.
                // When a context version below 3.0 is requested, implementations will
                // return the newest context version compatible with OpenGL versions less
                // than version 3.0.
                // GLX_CONTEXT_MAJOR_VERSION_ARB = 1
                context_attribs[1] = 1;
                // GLX_CONTEXT_MINOR_VERSION_ARB = 0
                context_attribs[3] = 0;

                ctxErrorOccurred = 0;

                printf( "Failed to create GL 3.0 context using old-style GLX context\n" );
                ctx = glXCreateContextAttribsARB( display, fbConfigs[0], 0, True, context_attribs );
            }
        }

        // clean up:
        XFree( fbConfigs );
        // Sync to ensure any errors generated are processed.
        XSync( display, False );

        // Restore the original error handler
        XSetErrorHandler( oldHandler );

        if ( ctxErrorOccurred || !ctx )
        {
            printf( "Failed to create an OpenGL context\n" );
            return 0;
        }

        // Verifying that context is a direct context
        if ( ! glXIsDirect ( display, ctx ) )
        {
            printf( "Indirect GLX rendering context obtained\n" );
        }
        else
        {
            printf( "Direct GLX rendering context obtained\n" );
        }

        printf( "Making context current\n" );

        whatToSwap=pbuffer;
        if ( !glXMakeContextCurrent( display, pbuffer, pbuffer, ctx ) )
        {
            fatalErrorGLX3(RED "glXMakeContextCurrent: Could not start rendering to pbuffer fbo" NORMAL);
            return 0;
        }

        if (!glXGetCurrentDrawable())
        {
            fatalErrorGLX3(RED "No drawable selected (pbuffer fbo)\n" NORMAL);

        }

        glClearColor( 0, 0.0, 0, 1 );
        glClear( GL_COLOR_BUFFER_BIT );
        glXSwapBuffers ( display, whatToSwap );

        printf( "GLX3.0 windowless context ready..\n" );
        return 1;
    }


    return start_glx3_stuffWindowed(WIDTH,HEIGHT,argc,argv);
}










int stop_glx3_stuff()
{
    glXMakeCurrent( display, 0, 0 );
    glXDestroyContext( display, ctx );

    /* In Pbuffer (headless) mode `win` and `cmap` are uninitialised — the
     * windowed code path never ran.  Calling XDestroyWindow / XFreeColormap
     * with a zero XID raises BadWindow / BadColormap.  Those errors are
     * harmless (cleanup is final-step) but they bubble up as a non-zero
     * exit status that scripts/video.sh interprets as "renderer crashed,
     * discard the JPEGs" — exactly the bug we're fixing here.  The Pbuffer
     * itself is owned by the GLXFBConfig and gets released when the
     * display is closed, so there's nothing else to free.  */
    if (!glx3_is_pbuffer) {
        XDestroyWindow( display, win );
        XFreeColormap( display, cmap );
    }
    XCloseDisplay( display );
    return 1;
}







int glx3_endRedraw()
{
    if (doubleBufferGLX3) glXSwapBuffers(display, whatToSwap);/* buffer swap does implicit glFlush */
    else glFlush();  /* explicit flush for single buffered case */
    return 1;
}


int glx3_should_close()
{
    return close_requested ? 1 : 0;
}

void glx3_request_close()
{
    close_requested = 1;
}

int glx3_checkEvents()
{
    /* Pbuffer / headless mode has no X event source (no window was mapped),
     * so XPending would be polling forever for events that can't arrive.
     * Just honour any close request and return. */
    if (glx3_is_pbuffer) return close_requested ? 0 : 1;

    /* If something already asked for a clean shutdown (Escape, ClientMessage
     * on a previous tick, an external call to glx3_request_close()), stop
     * pumping — the render loop's `while (glx3_checkEvents())` will exit. */
    if (close_requested) return 0;

    XEvent event;
    while (XPending(display))
    {
        XNextEvent(display, &event);
        switch (event.type)
        {
        case KeyPress:
        {
            KeySym keysym;
            char   buffer[1];
            int n = XLookupString((XKeyEvent*)&event, buffer, 1, &keysym, NULL);
            if (n == 1 && keysym == (KeySym)XK_Escape) {
                /* Escape used to be `exit(0)`, which skipped every dtor /
                 * fclose / dumpBVH the rest of the program had queued.  Now
                 * it asks for a clean shutdown so the main loop drops out
                 * and post-loop cleanup runs to completion. */
                glx3_request_close();
                return 0;
            }
            handleUserInput(keysym, 1, 0, 0);
            break;
        }
        case ButtonRelease:
        case ButtonPress:
            switch (event.xbutton.button)
            {
            case 1:
                handleUserInput(1, (event.type == ButtonPress),
                                event.xmotion.x_root, event.xmotion.y_root);
                break;
            case 2:
                handleUserInput(2, (event.type == ButtonPress),
                                event.xmotion.x_root, event.xmotion.y_root);
                break;
            case 3:
                handleUserInput(3, (event.type == ButtonPress),
                                event.xmotion.x_root, event.xmotion.y_root);
                break;
            }
            break;
        case ConfigureNotify:
            fprintf(stderr, "Received window configuration event..\n");
            windowSizeUpdated(event.xconfigure.width, event.xconfigure.height);
            /* fall through */
        case Expose:
            break;
        case ClientMessage:
            /* The window-manager close button arrives here when we've
             * registered for WM_DELETE_WINDOW (we did, just after creating
             * the window).  Without this case, the WM kills the X
             * connection on close and the next X call we make blows up
             * the process — which is exactly the "long render dies in
             * obscure X11 ways" failure mode the PLAN.md called out. */
            if (wm_delete_window != None &&
                (Atom)event.xclient.data.l[0] == wm_delete_window)
            {
                fprintf(stderr, "Received WM_DELETE_WINDOW — requesting clean shutdown\n");
                glx3_request_close();
                return 0;
            }
            break;
        case DestroyNotify:
            /* The window died from under us anyway (WM ignored the protocol,
             * or someone xkill'd us).  Treat as a close request so the loop
             * exits before we issue another X call on a dead window. */
            fprintf(stderr, "Received DestroyNotify — requesting clean shutdown\n");
            glx3_request_close();
            return 0;
        }
    }

    return close_requested ? 0 : 1;
}






