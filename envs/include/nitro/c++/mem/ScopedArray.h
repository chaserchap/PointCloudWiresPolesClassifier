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

#ifndef __MEM_SCOPED_ARRAY_H__
#define __MEM_SCOPED_ARRAY_H__

#include <cstddef>

namespace mem
{
    /*!
     *  \class ScopedArray
     *  \brief This class provides RAII for array allocations via new[].
     *         It is based on boost::scoped_array.
     */
    template <class T>
    class ScopedArray
    {
    public:
        typedef T ElementType;

        explicit ScopedArray(T* array = NULL) :
            mArray(array)
        {
        }

        ~ScopedArray()
        {
            delete[] mArray;
        }

        void reset(T* array = NULL)
        {
            delete[] mArray;
            mArray = array;
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
        ScopedArray(const ScopedArray& );
        const ScopedArray& operator=(const ScopedArray& );

    private:
        T* mArray;
    };
}

#endif
