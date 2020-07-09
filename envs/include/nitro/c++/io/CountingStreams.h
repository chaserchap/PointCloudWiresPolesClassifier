/* =========================================================================
 * This file is part of io-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * io-c++ is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public 
 * License along with this program; If not, 
 * see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef __IO_COUNTING_STREAMS_H__
#define __IO_COUNTING_STREAMS_H__

#include "io/ProxyStreams.h"

namespace io
{

/**
 * An OutputStream that keeps track of the number of bytes written to the stream.
 */
class CountingOutputStream: public ProxyOutputStream
{
public:
    CountingOutputStream(OutputStream *proxy, bool ownPtr = false) :
        ProxyOutputStream(proxy, ownPtr), mByteCount(0)
    {
    }
    virtual ~CountingOutputStream()
    {
    }

    using ProxyOutputStream::write;

    virtual void write(const sys::byte* b, sys::Size_T len)
    {
        ProxyOutputStream::write(b, len);
        mByteCount += len;
    }

    sys::Off_T getCount() const
    {
        return mByteCount;
    }

    void resetCount()
    {
        mByteCount = 0;
    }

protected:
    sys::Off_T mByteCount;
};

}

#endif
