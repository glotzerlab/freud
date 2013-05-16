//! Compute kernel for linear to sRGBA color conversion
/*! \param cmap flattened RGBA colormap
    \param start Start of the range to process
    \param end End of the range to process
    
    This is a low level kernel called by colorutil.cc. TBB threads are executed and in each worker thread evaluation
    this method is called to perform the actual color conversion on a range of colors in the colormap.
*/
export void viz_linearToSRGBA(uniform float cmap[],
                              uniform unsigned int start,
                              uniform unsigned int end)
    {
    // cmap is a flattend Nx4 float array, expand the start and end indices out to their flattened values
    start = start * 4;
    end = end * 4;
    
    // loop over all array entires
    foreach (i = start ... end)
        {
        // gamma correct RGB, but not A components
        if (i & 4 != 3)
            {
            cmap[i] = pow(cmap[i], 1.0f/2.2f);
            }
        }
    }
