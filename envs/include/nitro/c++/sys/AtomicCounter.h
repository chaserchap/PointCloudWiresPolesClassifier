/* =========================================================================
 * This file is part of sys-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * sys-c++ is free software; you can redistribute it and/or modify
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

#ifndef __SYS_ATOMIC_COUNTER_H__
#define __SYS_ATOMIC_COUNTER_H__

#if defined( __GNUC__ ) && ( defined( __i386__ ) || defined( __x86_64__ ) )
#include <sys/AtomicCounterX86.h>
#elif defined(WIN32)
#include <sys/AtomicCounterWin32.h>
#elif defined(__sun) && defined(HAVE_ATOMIC_H)
// atomic.h is available in Solaris 10+
// TODO: For Solaris 9 and older, we currently use the mutex implementation
//       http://blogs.oracle.com/d/entry/atomic_operations
//       provides a snippet of assembly code for atomic incrementing in
//       Solaris 9 - this would be a starting point if a faster implementation
//       is needed
#include <sys/AtomicCounterSolaris.h>
#else
// Bummer - need to fall back on a slow mutex implementation
#include "AtomicCounterMutex.h"
#endif

namespace sys
{
/*!
 *  \class AtomicCounter
 *  \brief This class provides atomic incrementing, decrementing, and setting
 *         of an unsigned integer.  All operations are thread-safe.
 *
 *  TODO: Currently, we use the ValueType typedef for whatever the underlying
 *        integer type is.  Should we instead provide implementations that are
 *        explicitly for 32 bit and 64 bit integers?  For Solaris and Windows,
 *        there are equivalent instructions so this would be simple.  For X86,
 *        would need to research the assembly instructions to find the
 *        equivalent 64-bit instruction.
 *
 *  TODO: Provide other operations such as getThenSet() and compareThenSet().
 */
class AtomicCounter
{
public:
    typedef AtomicCounterImpl::ValueType ValueType;

    //! Constructor
    AtomicCounter(ValueType initialValue = 0) :
        mImpl(initialValue)
    {
    }

    /*!
     *   Increment the value
     *   \return The value PRIOR to incrementing
     */
    ValueType getThenIncrement()
    {
        return mImpl.getThenIncrement();
    }

    ValueType operator++(int )
    {
        return getThenIncrement();
    }

    /*!
     *   Increment the value
     *   \return The value AFTER incrementing
     */
    ValueType incrementThenGet()
    {
        return (getThenIncrement() + 1);
    }

    ValueType operator++()
    {
        return incrementThenGet();
    }

    //! Increment the value
    void increment()
    {
        getThenIncrement();
    }

    /*!
     *   Decrement the value
     *   \return The value PRIOR to decrementing
     */
    ValueType getThenDecrement()
    {
        return mImpl.getThenDecrement();
    }

    ValueType operator--(int )
    {
        return getThenDecrement();
    }

    /*!
     *   Decrement the value
     *   \return The value AFTER decrementing
     */
    ValueType decrementThenGet()
    {
        return (getThenDecrement() - 1);
    }

    ValueType operator--()
    {
        return decrementThenGet();
    }

    //! Decrement the value
    void decrement()
    {
        getThenDecrement();
    }

    /*!
     *   Get the current value
     *   \return The current value
     */
    ValueType get() const
    {
        return mImpl.get();
    }

private:
    // Noncopyable
    AtomicCounter(const AtomicCounter& );
    const AtomicCounter& operator=(const AtomicCounter& );

private:
    AtomicCounterImpl mImpl;
};
}

#endif
