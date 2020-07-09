/* =========================================================================
 * This file is part of sys-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2012, General Dynamics - Advanced Information Systems
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

#ifndef __SYS_UTC_DATE_TIME_H__
#define __SYS_UTC_DATE_TIME_H__

#include "DateTime.h"
#include "Export.h"

namespace sys
{

/*!
 *  Representation of a UTC date/time structure.
 */
class DLL_PUBLIC_CLASS UTCDateTime : public DateTime
{
protected:
    /**
     * @brief Set the millis value from the members
     */
    virtual void toMillis();

    //! Given seconds since the epoch, provides the UTC time
    virtual void getTime(time_t numSecondsSinceEpoch, tm& t) const;

public:
    static const char DEFAULT_DATETIME_FORMAT[];

    /*!
     *  Construct as current date and time (UTC).
     */
    UTCDateTime();

    /*!
     *  Construct with time values.  Date will be today.
     */
    UTCDateTime(int hour, int minute, double second);
    /*!
     *  Construct with date values.  Time will be 00:00:00.
     */
    UTCDateTime(int year, int month, int day);
    /*!
     *  Construct with date and time values.
     */
    UTCDateTime(int year, int month, int day, 
            int hour, int minute, double second);
    /*!
     *  Construct with time in milliseconds.
     */
    UTCDateTime(double timeInMillis);

    // unhide in the base class format method
    using DateTime::format;

    /*!
     *  The default formatting looks like this:
     *  %Y-%m-%dT%H:%M:%SZ
     *  2011-10-19T11:59:46Z
     */
    std::string format() const;

};

}

#endif//__SYS_UTC_DATE_TIME_H__
