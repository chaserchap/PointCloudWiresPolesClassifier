/* =========================================================================
 * This file is part of mem-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * mem-c++ is free software; you can redistribute it and/or modify
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

#ifndef __MEM_SCOPED_ALIGNED_ARRAY_H__
#define __MEM_SCOPED_ALIGNED_ARRAY_H__

#include <cstddef>

#include <sys/Conf.h>

namespace mem
{
    /*!
     *  \class ScopedAlignedArray
     *  \brief This class provides RAII for alignedAlloc() and alignedFree()
     */
    template <class T>
    class ScopedAlignedArray
    {
    public:
        typedef T ElementType;

        explicit ScopedAlignedArray(size_t numElements = 0) :
            mArray(allocate(numElements))
        {
        }

        ~ScopedAlignedArray()
        {
            if (mArray)
            {
                // Don't expect sys::alignedFree() would ever throw, but just
                // in case...
                try
                {
                    sys::alignedFree(mArray);
                }
                catch (...)
                {
                }
            }
        }

        void reset(size_t numElements = 0)
        {
            if (mArray)
            {
                sys::alignedFree(mArray);
                mArray = NULL;
            }

            mArray = allocate(numElements);
        }

        T& operator[](std::ptrdiff_t idx) const
        {
            return mArray[idx];
        }

        T* get() const
        {
            return mArray;
        }

        T* release()
        {
            T* const array = mArray;
            mArray = NULL;
            return array;
        }

    private:
        // Noncopyable
        ScopedAlignedArray(const ScopedAlignedArray& );
        const ScopedAlignedArray& operator=(const ScopedAlignedArray& );

        static
        T* allocate(size_t numElements)
        {
            if (numElements > 0)
            {
                const size_t numBytes(numElements * sizeof(T));
                return static_cast<T *>(sys::alignedAlloc(numBytes));
            }
            else
            {
                return NULL;
            }
        }

    private:
        T* mArray;
    };
}

#endif
