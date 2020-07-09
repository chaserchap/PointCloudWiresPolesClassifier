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


#ifndef __SYS_TIME_STAMP_H__
#define __SYS_TIME_STAMP_H__

#include <string>
#include "str/Format.h"
#include "Conf.h"
#include "LocalDateTime.h"
#include "UTCDateTime.h"
#include "Export.h"

/*!
 *  \file  TimeStamp.h
 *  \brief Get a timestamp in a system-independent manner
 * 
 *  Provide the API for timestamps
 */

namespace sys
{
/*!
 *  \class TimeStamp
 *  \brief Class for timestamping
 *
 */
class DLL_PUBLIC_CLASS TimeStamp
{
public:
    //! The maximum length of a timestamp
    enum { MAX_TIME_STAMP = 64 };

    /*!
     *  Default constructor.
     *  \param is24HourTime Optional specifier for military time
     */
    TimeStamp(bool is24HourTime = false) : m24HourTime(is24HourTime)
    {}
    //!  Destructor
    ~TimeStamp()
    {}

    /*!
     *  Produces a local-time string timestamp
     *  \return local-time
     */
    std::string local() const
    {
        sys::LocalDateTime dt;
        return dt.format(getFormat());
    }

    /*!
     *  Produces a gmt string timestamp
     *  \return gmt
     */
    std::string gmt() const
    {
        sys::UTCDateTime dt;
        return dt.format(getFormat());
    }

private:
    /*!
     *  Sets up string in desired format
     *  \return The string format for printing
     */
    const char* getFormat() const
    {
        if (m24HourTime)
        {
            return "%m/%d/%Y, %H:%M:%S";
        }
        return "%m/%d/%Y, %I:%M:%S%p";
    }

    //!  Military time???
    bool m24HourTime;
};

}

#endif
