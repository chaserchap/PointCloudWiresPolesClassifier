/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2010, General Dynamics - Advanced Information Systems
 *
 * NITRO is free software; you can redistribute it and/or modify
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
 * License along with this program; if not, If not,
 * see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef __NITF_BUFFERED_WRITER_HPP__
#define __NITF_BUFFERED_WRITER_HPP__

#include <sys/File.h>
#include <mem/ScopedArray.h>
#include <nitf/CustomIO.hpp>
#include <sys/Export.h>

namespace nitf
{
class DLL_PUBLIC_CLASS BufferedWriter : public CustomIO
{
public:
    BufferedWriter(const std::string& file, size_t bufferSize);

    BufferedWriter(const std::string& file,
                   char* buffer,
                   size_t size,
                   bool adopt = false);

    void flushBuffer();

    nitf::Uint64 getTotalWritten() const
    {
        return mTotalWritten;
    }

    nitf::Uint64 getNumBlocksWritten() const
    {
        return mBlocksWritten;
    }

    nitf::Uint64 getNumPartialBlocksWritten() const
    {
        return mPartialBlocks;
    }

protected:
    virtual void readImpl(char* buf, size_t size);

    virtual void writeImpl(const char* buf, size_t size);

    virtual bool canSeekImpl() const;

    virtual nitf::Off seekImpl(nitf::Off offset, int whence);

    virtual nitf::Off tellImpl() const;

    virtual nitf::Off getSizeImpl() const;

    virtual int getModeImpl() const;

    virtual void closeImpl();

private:
    const size_t mBufferSize;
    const mem::ScopedArray<char> mScopedBuffer;
    char* const mBuffer;

    nitf::Uint64 mPosition;
    nitf::Uint64 mTotalWritten;
    nitf::Uint64 mBlocksWritten;
    nitf::Uint64 mPartialBlocks;

    // NOTE: This is at the end to give us a chance to adopt the buffer
    //       in ScopedArray in case sys::File's constructor throws
    mutable sys::File mFile;
};

}
#endif
