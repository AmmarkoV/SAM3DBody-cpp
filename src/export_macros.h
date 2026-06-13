#pragma once

#ifdef _WIN32
    #ifdef fast_sam_3dbody_EXPORTS
        #define FSB_API __declspec(dllexport)
    #else
        #define FSB_API __declspec(dllimport)
    #endif
#else
    #define FSB_API
#endif
